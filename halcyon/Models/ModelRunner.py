import logging
from typing import Callable, List, Tuple
import tempfile
from pathlib import Path
from collections.abc import Generator

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from halcyon.Datasets.ExperimentCondition import ExperimentCondition
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.Datasets.ExpressionData import ExpressionData
from halcyon.Models.Losses import MSELossFlexible
from halcyon.Models.HybridModel import HybridModel
from halcyon.Models.Model import GeneExpressionModel
from halcyon.Simulators.SimulationResult import SimulationResult
from halcyon.Results.TidyDfResults import TidyDfResults


class ModelRunner:
    mlflow_prediction_folder_name = "evaluation"

    def __init__(
        self,
        model: GeneExpressionModel,
        nr_epochs: int,
        batch_size: int = 32,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs=None,
        loss: nn.Module = MSELossFlexible(),
        file_name_prefix="trainer_",
    ):
        """Class to handle running models on simulation results. Handles the integration between dataloader and model.

        Args:
            model (Model): Model to be trained
            nr_epochs (int): How many epochs to run the training
            batch_size (int, optional): Batch size to use during training. Defaults to 32.
            optimizer_fn (Callable, optional): Optimizer function. Defaults to torch.optim.Adam.
            optimizer_kwargs (dict | None, optional): Kwargs to pass to optimiser (e.g. learning rate). Defaults to None.
            loss (Callable, optional): Loss functions. Defaults to mse loss.
        """
        self.model = model
        self.nr_epochs = nr_epochs
        self.batch_size = batch_size
        self.loss_fn = loss

        self.file_name_prefix = file_name_prefix
        self.file_name_suffix = ".pt"

        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        self.optimizer = optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        self.model.to(self.model.device)

        # Loss history after training. Is empty initially (because no training has been performed yet).
        self.loss_history = []

    def __repr__(self):
        return (
            f"Modelrunner (Model={self.model},\n optimizer={self.optimizer.__class__.__name__},\n "
            f"optimizer_kwargs={self.optimizer_kwargs},\n nr_epochs={self.nr_epochs},\n batch_size={self.batch_size}\n)"
        )

    def get_hyperparams_to_log(self) -> dict:
        """Get both hyperparameters of the model and the optimisation to log to e.g. mlflow

        Returns:
            dict: with key as hyperparameter name and value as hyperparameter value
        """
        out_dict = {}
        for hyperparam_name, hyperparam_value in self.model.get_hyperparams():
            if hyperparam_name == "device":
                continue
            if torch.is_tensor(hyperparam_value):
                hyperparam_value = hyperparam_value.cpu().item()
            out_dict[hyperparam_name] = hyperparam_value

        if self.optimizer_kwargs:
            for kwarg_name, value in self.optimizer_kwargs.items():
                out_dict[kwarg_name] = value

        # Add the ModelRunner-level hyperparams
        out_dict.update(
            {
                "nr_epochs": self.nr_epochs,
                "batch_size": self.batch_size,
                "loss_fn": self.loss_fn.__class__.__name__,
                "model_class": self.model.name,
            }
        )
        return out_dict

    @property
    def file_name(self) -> str:
        file_name = f"{self.file_name_prefix}{self.model.name}"
        for hyperparam_name, hyperparam_value in self.get_hyperparams_to_log().items():
            file_name += f"_{hyperparam_name}_{hyperparam_value}"
        file_name += self.file_name_suffix
        return file_name

    def train(self, dataset: ExpressionDataWithConditions):
        """Perform model training."""
        dataloader = self._dataloader_factory(dataset)
        time_points_tensor = self._get_time_points_as_tensor(dataset)

        for epoch in range(self.nr_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                self.optimizer.zero_grad()
                ground_truth_expression, prediction = (
                    self._predict_expression_from_batch(time_points_tensor, batch)
                )
                loss = self.loss_fn(prediction, ground_truth_expression, self.model)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            mean_epoch_loss = epoch_loss / len(dataloader)
            self.loss_history.append(mean_epoch_loss)
            logging.info(f"Epoch {epoch}, loss {mean_epoch_loss}")

    @torch.no_grad()
    def predict(
        self, dataset: ExpressionDataWithConditions
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Use model to predict the knockouts in the provided dataset

        Args:
            dataset (ExpressionDataWithConditions): Dataset on which to predict.

        Returns:
            Tuple of three items: list of conditions (as strings), tensor of shape (n_conditions, n_genes, n_timepoints) with ground truth expression data, tensor of shape (n_conditions, n_genes, n_timepoints) with predicted expression data.
        """
        self.model.eval()
        dataloader = self._dataloader_factory(dataset)
        time_points_tensor = self._get_time_points_as_tensor(dataset)

        experiment_as_string_list = []
        batches_gt_expression_data = []
        all_predictions = []
        for batch in tqdm(dataloader):
            ground_truth_expression, prediction = self._predict_expression_from_batch(
                time_points_tensor, batch
            )
            experiment_as_string_list.extend(batch[2])
            batches_gt_expression_data.append(ground_truth_expression.cpu())
            all_predictions.append(prediction.detach().cpu())
        self.model.train()

        # Concat batches together into one big tensor
        all_gt_expression_data = torch.cat(batches_gt_expression_data, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        return experiment_as_string_list, all_gt_expression_data, all_predictions

    def _predict_expression_from_batch(
        self, time_points_tensor: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a batch from the dataloader, use the model to predict the expression

        Args:
            time_points_tensor (torch.Tensor): points at which to evaluate the ODE integration
            batch (tuple[torch.Tensor, torch.Tensor]): The batch from the dataloader which contains the experiment data
              (to set e.g. knockout) and ground_truth_expression (which are needed for the intial condition)

        Returns:
            Tuple of two tensors: ground_truth_expressions (used for loss calculation) and the model prediction
        """
        experiment_data, ground_truth_expression, _ = batch
        experiment_data = experiment_data.to(self.model.device)
        ground_truth_expression = ground_truth_expression.to(self.model.device)

        self.model.set_conditions(experiment_data)

        prediction = self.model.integrate(
            initial_state=ground_truth_expression[:, :, 0],
            t_eval=time_points_tensor,
        )

        return ground_truth_expression, prediction

    def train_and_predict_for_mlflow(
        self,
        dataset_dict: dict[str, ExpressionDataWithConditions],
        train_data_key_in_dict: str,
        simulation_parent_id: str,
        do_mlflow_model_logging: bool = False,
    ):
        """Carry out training and predicting and log everything to MLFlow

        Args:
            dataset_dict (dict[str, ExpressionDataWithConditions]): Dictionary with dataset name as key and dataset as value.
                The model will be trained on the dataset specified by train_data_key_in_dict and evaluated on all datasets in the dictionary.
            train_data_key_in_dict (str): Key that specifies on which dataset in the dataset_dict to train the model.
            simulation_parent_id (str, optional): Simulation parent run ID of the run from which the data was
                generated. Used in MLFlow for data lineage.
            do_mlflow_model_logging (bool, optional): If set to True, the model will be logged to MLFlow using the
              pytorch.log_model function. OTherwise will just save the .pt file as artifact.
              Defaults to False.
        """
        with mlflow.start_run(
            parent_run_id=simulation_parent_id, run_name=self.model.name
        ):
            mlflow.set_tag("run_type", "model_training_and_prediction")
            mlflow.log_params(self.get_hyperparams_to_log())
            self.train(dataset_dict[train_data_key_in_dict])
            self._log_train_losses_to_mlflow()

            # Evaluate model on both datasets and log raw predictions
            for dataset_name, dataset in dataset_dict.items():
                experiment_as_string_list, all_gt_expression_data, all_predictions = (
                    self.predict(dataset)
                )
                final_loss = self.loss_fn(
                    all_predictions, all_gt_expression_data, self.model
                )
                mlflow.log_metric(f"{dataset_name}_final_loss", final_loss.item())

                with tempfile.TemporaryDirectory() as tmpdir:
                    logging.debug("Temp dir:", tmpdir)
                    tmpdir_path = Path(tmpdir)

                    # Also save the model as .pt just to be sure of the model logging
                    model_save_path = tmpdir_path / self.file_name
                    self.save(model_save_path)
                    mlflow.log_artifact(model_save_path)

                    predicted_file_path = self._save_predictions_to_file(
                        dataset_name,
                        dataset,
                        experiment_as_string_list,
                        all_gt_expression_data,
                        all_predictions,
                        tmpdir_path,
                    )

                    mlflow.log_artifact(
                        predicted_file_path,
                        artifact_path=self.mlflow_prediction_folder_name,
                    )

            if do_mlflow_model_logging:
                mlflow.pytorch.log_model(self.model, name=self.model.name)

    def _save_predictions_to_file(
        self,
        dataset_name: str,
        dataset: ExpressionDataWithConditions,
        experiment_as_string_list: List[str],
        all_gt_expression_data: torch.Tensor,
        all_predictions: torch.Tensor,
        tmpdir_path: Path,
    ):
        """Save the raw prediction tensors etc to a parquet file to log as artifact in MLFlow"""
        predicted_file_path = tmpdir_path / f"prediction_{dataset_name}_data.parquet"

        prediction_df = self._prediction_outputs_to_df(
            (
                experiment_as_string_list,
                all_gt_expression_data,
                all_predictions,
            ),
            dataset.get_time_points(),
            modelled_gene_labels=self.model.modelled_gene_labels,
        )
        prediction_df.to_parquet(predicted_file_path, index=False)
        return predicted_file_path

    def _prediction_outputs_to_df(
        self,
        prediction_outputs: Tuple[List[str], torch.Tensor, torch.Tensor],
        time_points: np.ndarray,
        modelled_gene_labels: List[str],
    ) -> pd.DataFrame:
        """Converts the output of a prediction (list of conditions, tensor of ground truth expression data,
          tensor of predicted expression data) into a long format dataframe that can be easily plotted.

        Args:
            prediction_outputs (Tuple[List[str], torch.Tensor, torch.Tensor]): Output of the predict.
                First item is list of conditions,
                second item is tensor of shape (n_conditions, n_genes, n_timepoints) with ground truth expression data,
                third item is tensor of shape (n_conditions, n_genes, n_timepoints) with predicted expression data.
            dataset (ExpressionDataWithConditions): Dataset to use to store name in dataframe and to get time points from
            modelled_gene_labels (List[str]): Genes that the model models

        Yields:
            pd.DataFrame: The long format dataframe.
        """
        df_list = []

        for condition, gt_matrix, pred_matrix in zip(*prediction_outputs):
            melted_expression_df = []
            for matrix, col_label in zip(
                [pred_matrix, gt_matrix],
                [TidyDfResults.EXPRESSION_COL, TidyDfResults.GROUND_TRUTH_COL],
            ):
                long_form_df = pd.DataFrame(
                    matrix,
                    columns=time_points,
                    index=pd.Index(modelled_gene_labels, name=TidyDfResults.GENE_COL),
                )
                melted_df = long_form_df.melt(
                    ignore_index=False,
                    var_name=TidyDfResults.TIME_COL,
                    value_name=col_label,
                ).set_index(TidyDfResults.TIME_COL, append=True)
                melted_expression_df.append(melted_df)

            one_condition_df = pd.concat(melted_expression_df, axis=1).reset_index()

            one_condition_df[TidyDfResults.CONDITION_COL] = condition
            df_list.append(one_condition_df)

        df = pd.concat(df_list)
        return df

    def _log_train_losses_to_mlflow(self):
        """After training, this function can be used to log the loss history to MLFlow"""
        for epoch, loss in enumerate(self.loss_history):
            mlflow.log_metric("train_loss", loss, step=epoch)

    def save(self, checkpoint_path: Path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No existing model found at {checkpoint_path}.")

        logging.info(f"Loading from {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.model.device, weights_only=False
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_history = checkpoint["loss_history"]
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            # Get rid of some weird keys that might be saved in this model
            new_dict = {}
            for key, value in checkpoint["model_state_dict"].items():
                possible_beginnings_to_skip = [
                    "_mm._synthesis_module",
                    "_mm._decay_module",
                    "_synthesis_module",
                    "_decay_module",
                ]
                for beginning_to_skip in possible_beginnings_to_skip:
                    if key.startswith(beginning_to_skip):
                        break
                else:
                    new_dict[key] = value

            self.model.load_state_dict(new_dict)

    def _get_time_points_as_tensor(
        self, dataset: ExpressionDataWithConditions
    ) -> torch.Tensor:
        """Get timepoints of the dataset to know at which point to evaluate the ODE integration"""
        time_points = dataset._get_time_points()
        time_points_tensor = torch.tensor(
            time_points, dtype=torch.float32, device=self.model.device
        )
        return time_points_tensor

    def _dataloader_factory(self, dataset: ExpressionDataWithConditions) -> DataLoader:
        """Matches dataset with genes in model and returns dataloader to be used for training or inference"""
        dataset = dataset.match_genes_to_list(self.model.modelled_gene_labels)
        assert (
            dataset.genes == self.model.modelled_gene_labels
        ), "Gene labels do not match"
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )


def run_training_job(
    job: ModelRunner,
    dataset_dict: dict[str, ExpressionDataWithConditions],
    train_data_key_in_dict: str,
    simulation_parent_id: str = None,
    do_mlflow_model_logging: bool = False,
):
    """Runs one training job. Can be called from mp.Pool to run multiple in parallel"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(message)s",
        force=True,
    )
    try:
        output = job.train_and_predict_for_mlflow(
            dataset_dict,
            train_data_key_in_dict,
            simulation_parent_id,
            do_mlflow_model_logging,
        )
    except AssertionError as e:
        logging.error(f"Error occurred while running job: {e}")
        output = {"error": str(e)}
    return output
