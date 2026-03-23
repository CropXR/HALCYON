from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.ModelRunner.EarlyStopper import EarlyStopper
from halcyon.Models.GeneExpressionModel import GeneExpressionModel
from halcyon.Models.Losses import MSELossFlexible
from halcyon.results.DfColumnNames import DfColumnNames


import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


import logging
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple


class ModelRunner:
    mlflow_prediction_folder_name = "evaluation"

    def __init__(
        self,
        model: GeneExpressionModel,
        batch_size: int = 32,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs=None,
        loss: nn.Module = MSELossFlexible(),
        file_name_prefix="trainer_",
    ):
        """Class to handle running models on simulation results. Handles the integration between dataloader and model.

        Args:
            model (Model): Model to be trained
            batch_size (int, optional): Batch size to use during training. Defaults to 32.
            optimizer_fn (Callable, optional): Optimizer function. Defaults to torch.optim.Adam.
            optimizer_kwargs (dict | None, optional): Kwargs to pass to optimiser (e.g. learning rate). Defaults to None.
            loss (Callable, optional): Loss function. Defaults to mse loss.
        """
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = loss
        self.mse_loss_fn = MSELossFlexible()

        self.file_name_prefix = file_name_prefix
        self.file_name_suffix = ".pt"

        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        self.optimizer = optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        self.early_stopper = EarlyStopper()

        self.model.to(self.model.device)

        # Loss history after training. Is empty initially (because no training has been performed yet).
        self.loss_history = []

    def __repr__(self):
        return (
            f"Modelrunner (Model={self.model},\n optimizer={self.optimizer.__class__.__name__},\n "
            f"optimizer_kwargs={self.optimizer_kwargs},\n batch_size={self.batch_size},\n loss_fn={self.loss_fn}\n)"
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

        if self.early_stopper:
            for kwarg_name, value in self.early_stopper.hyperparams_to_log().items():
                out_dict[kwarg_name] = value

        # Add the ModelRunner-level hyperparams
        out_dict.update(
            {
                "batch_size": self.batch_size,
                "loss_fn": str(self.loss_fn),
                "model_class": self.model.name,
            }
        )
        return out_dict

    @property
    def file_name(self) -> str:
        file_name = f"{self.file_name_prefix}{self.model.name}"
        for hyperparam_name, hyperparam_value in self.get_hyperparams_to_log().items():
            file_name += f"_{hyperparam_name}_{hyperparam_value}"
        return file_name

    def train(
        self,
        train_dataset: ExpressionDataWithConditions,
        nr_epochs: int,
        val_dataset: ExpressionDataWithConditions | None = None,
        test_dataset: ExpressionDataWithConditions | None = None,
        save_every_n_epochs: int | None = None,
    ):
        """Perform model training.

        Args:
            train_dataset (ExpressionDataWithConditions): Dataset to train on
            val_dataset (ExpressionDataWithConditions): Dataset to report val_loss on
            test_dataset (ExpressionDataWithConditions): Dataset to report test_loss on
            save_every_n_epochs (int | None, optional): If provided, save model every
              n_epochs and log it to MLFLow, ensures you can still retrieve your
              model if training crashes prematurely. Defaults to None, i.e. no saving.
        """
        train_dataloader = self._dataloader_factory(train_dataset)
        time_points_tensor = self._get_time_points_as_tensor(train_dataset)
        for epoch in range(nr_epochs):
            self.model.train()
            epoch_loss = 0.0
            mse_loss = 0.0
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                ground_truth_expression, prediction = (
                    self._predict_expression_from_batch(time_points_tensor, batch)
                )
                loss = self.loss_fn(prediction, ground_truth_expression, self.model)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                with torch.no_grad():
                    mse_loss += self.mse_loss_fn(
                        prediction, ground_truth_expression, self.model
                    ).item()

            mean_epoch_loss = epoch_loss / len(train_dataloader)
            mean_mse_loss = mse_loss / len(train_dataloader)
            logging.info(f"Epoch {epoch}, train_loss\t{mean_epoch_loss}")

            self.loss_history.append(mean_epoch_loss)
            if mlflow.active_run():
                mlflow.log_metric("train_loss", mean_epoch_loss, step=epoch)
                mlflow.log_metric("train_mse", mean_mse_loss, step=epoch)

            if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                self._log_model_to_mlflow(epoch)

            # --- val/test ---
            if val_dataset is not None:
                self._log_loss_on_unseen_data(
                    val_dataset, "val_loss", epoch, self.loss_fn
                )
                val_mse = self._log_loss_on_unseen_data(
                    val_dataset, "val_mse", epoch, self.mse_loss_fn
                )

            if test_dataset is not None:
                self._log_loss_on_unseen_data(
                    test_dataset, "test_loss", epoch, self.loss_fn
                )
                self._log_loss_on_unseen_data(
                    test_dataset, "test_mse", epoch, self.mse_loss_fn
                )

            if (
                self.early_stopper
                and val_dataset is not None
                and self.early_stopper.should_stop(val_mse)
            ):
                logging.info(f"Early stopping at epoch {epoch}")
                self._log_model_if_not_saved_already(
                    save_every_n_epochs, epoch, total_nr_epochs=nr_epochs
                )
                break
        else:
            # Save model when training finishes
            self._log_model_if_not_saved_already(
                save_every_n_epochs, epoch, total_nr_epochs=nr_epochs
            )

    def _log_model_if_not_saved_already(
        self, save_every_n_epochs: int, current_epoch: int, total_nr_epochs: int
    ):
        """Log final model state, unless it was already saved at this epoch via save_every_n_epochs."""
        if not (save_every_n_epochs and total_nr_epochs % save_every_n_epochs == 0):
            self._log_model_to_mlflow(current_epoch)

    @torch.no_grad()
    def _log_loss_on_unseen_data(
        self,
        dataset: ExpressionDataWithConditions,
        loss_name: str,
        epoch: int,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Calculate loss on unseen data (e.g. validation or test data) and log to mlflow if available

        Args:
            dataset (ExpressionDataWithConditions): Dataset on which to calculate loss
            loss_name (str): Name of loss that should be reported to MLFlow
            epoch (int): Current epoch
            loss_fn (nn.Module): Loss function to use

        Returns:
            Loss as toch.tensor
        """
        _, gt_expression, predicted_expression = self.predict(
            dataset, disable_progress_bar=True
        )
        loss = loss_fn(predicted_expression, gt_expression, self.model)
        logging.info(f"Epoch {epoch}, {loss_name}\t{loss}")
        if mlflow.active_run():
            mlflow.log_metric(loss_name, loss, step=epoch)
        return loss

    def _log_model_to_mlflow(self, epoch: int):
        if mlflow.active_run():
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / f"{self.file_name}_epoch{epoch + 1}.pt"
                self.save(checkpoint_path)
                mlflow.log_artifact(checkpoint_path, artifact_path="model")

    @torch.no_grad()
    def predict(
        self, dataset: ExpressionDataWithConditions, disable_progress_bar: bool = False
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Use model to predict the knockouts in the provided dataset

        Args:
            dataset (ExpressionDataWithConditions): Dataset on which to predict.

        Returns:
            Tuple of three items: list of conditions (as strings), tensor of shape (n_conditions, n_genes, n_timepoints) with ground truth expression data,
            tensor of shape (n_conditions, n_genes, n_timepoints) with predicted expression data.
        """
        # set model evaluation mode to ensure consistent behavior during prediction, e.g. with Dropout Layers or Batch Normalization Layers
        self.model.eval()
        dataloader = self._dataloader_factory(dataset)
        time_points_tensor = self._get_time_points_as_tensor(dataset)

        experiment_as_string_list = []
        batches_gt_expression_data = []
        all_predictions = []
        for batch in tqdm(dataloader, disable=disable_progress_bar):
            ground_truth_expression, prediction = self._predict_expression_from_batch(
                time_points_tensor, batch
            )
            experiment_as_string_list.extend(batch[2])
            batches_gt_expression_data.append(ground_truth_expression.cpu())
            all_predictions.append(prediction.detach().cpu())
        # set model back to training mode
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
        simulation_parent_id: str,
        train_data_key_in_dict: str,
        nr_epochs: int,
        val_data_key_in_dict: str | None = None,
        test_data_key_in_dict: str | None = None,
        do_mlflow_model_logging: bool = False,
        save_every_n_epochs: int | None = 5,
    ):
        """Carry out training and predicting and log everything to MLFlow

        Args:
            dataset_dict (dict[str, ExpressionDataWithConditions]): Dictionary with dataset name as key and dataset as value.
                The model will be trained on the dataset specified by train_data_key_in_dict and evaluated on all datasets in the dictionary.
            train_data_key_in_dict (str): Key that specifies on which dataset in the dataset_dict to train the model.
            simulation_parent_id (str): Simulation parent run ID of the run from which the data was
                generated. Used in MLFlow for data lineage.
            do_mlflow_model_logging (bool, optional): If set to True, the model will be logged to MLFlow using the
              pytorch.log_model function. OTherwise will just save the .pt file as artifact.
              Defaults to False.
            save_every_n_epochs (int | None, optional): If provided, save model to
              MLFLow every n_epochs so it can still be retrieved if training crashes prematurely.
        """
        with mlflow.start_run(
            parent_run_id=simulation_parent_id, run_name=self.model.name
        ):
            mlflow.set_tag("run_type", "model_training_and_prediction")
            mlflow.log_params(self.get_hyperparams_to_log())
            mlflow.log_param("nr_epochs", nr_epochs)

            log_path = self._setup_logging_to_file()
            try:
                self.train(
                    train_dataset=dataset_dict[train_data_key_in_dict],
                    nr_epochs=nr_epochs,
                    val_dataset=dataset_dict.get(val_data_key_in_dict),
                    test_dataset=dataset_dict.get(test_data_key_in_dict),
                    save_every_n_epochs=save_every_n_epochs,
                )

                # Finally: evaluate model on all datasets and log raw predictions
                for dataset_name, dataset in dataset_dict.items():
                    with tempfile.TemporaryDirectory() as tmpdir:
                        logging.debug(f"Temp dir: {tmpdir}")
                        tmpdir_path = Path(tmpdir)

                        final_loss, mse_loss, predicted_file_path = (
                            self._predict_for_mlflow_logging(
                                dataset_name, dataset, tmpdir_path
                            )
                        )
                        mlflow.log_metric(
                            f"{dataset_name}_final_loss", final_loss.item()
                        )
                        mlflow.log_metric(
                            f"{dataset_name}_final_mse_loss", mse_loss.item()
                        )
                        mlflow.log_artifact(
                            predicted_file_path,
                            artifact_path=self.mlflow_prediction_folder_name,
                        )

                if do_mlflow_model_logging:
                    mlflow.pytorch.log_model(self.model, name=self.model.name)
            finally:
                mlflow.log_artifact(log_path)

    def _setup_logging_to_file(self) -> Path:
        """Make run log to file so errors can be easily found back"""
        log_path = Path(f"logs/{mlflow.active_run().info.run_id}.log")
        log_path.parent.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(processName)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(),
            ],
            force=True,
        )

        return log_path

    def _predict_for_mlflow_logging(
        self,
        dataset_name: str,
        dataset: ExpressionDataWithConditions,
        tmpdir_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor, Path]:
        """Creates the files and metrics of model predictions that are logged to MLFlow.

        Returns:
            Tuple of final_loss, mse_loss, and file_path to model predictions saved as .parquet
        """
        experiment_as_string_list, all_gt_expression_data, all_predictions = (
            self.predict(dataset)
        )
        final_loss = self.loss_fn(all_predictions, all_gt_expression_data, self.model)
        final_mse_loss = self.mse_loss_fn(
            all_predictions, all_gt_expression_data, self.model
        )

        predicted_file_path = self._save_predictions_to_file(
            dataset_name,
            dataset,
            experiment_as_string_list,
            all_gt_expression_data,
            all_predictions,
            tmpdir_path,
        )

        return final_loss, final_mse_loss, predicted_file_path

    def _save_predictions_to_file(
        self,
        dataset_name: str,
        dataset: ExpressionDataWithConditions,
        experiment_as_string_list: List[str],
        all_gt_expression_data: torch.Tensor,
        all_predictions: torch.Tensor,
        tmpdir_path: Path,
    ) -> Path:
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
                [DfColumnNames.EXPRESSION_COL, DfColumnNames.GROUND_TRUTH_COL],
            ):
                long_form_df = pd.DataFrame(
                    matrix,
                    columns=time_points,
                    index=pd.Index(modelled_gene_labels, name=DfColumnNames.GENE_COL),
                )
                melted_df = long_form_df.melt(
                    ignore_index=False,
                    var_name=DfColumnNames.TIME_COL,
                    value_name=col_label,
                ).set_index(DfColumnNames.TIME_COL, append=True)
                melted_expression_df.append(melted_df)

            one_condition_df = pd.concat(melted_expression_df, axis=1).reset_index()

            one_condition_df[DfColumnNames.CONDITION_COL] = condition
            df_list.append(one_condition_df)

        df = pd.concat(df_list)
        return df

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
