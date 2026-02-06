import logging
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from halcyon.Datasets.ExperimentCondition import ExperimentCondition
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions

from typing import Callable
from halcyon.Datasets.ExpressionData import ExpressionData
from halcyon.Losses import MSELossFlexible
from halcyon.Models.HybridModel import HybridModel
from halcyon.Models.Model import GeneExpressionModel
import torch.nn as nn
import seaborn as sns
from halcyon.Simulators.SimulationResult import SimulationResult
from tqdm import tqdm


class ModelRunner:
    def __init__(
        self,
        model: GeneExpressionModel,
        nr_epochs: int,
        save_dir: Path,
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

        file_name = self.get_save_file_name()
        self.save_path = save_dir / file_name

        # Loss history after training. Is empty initially (because no training has been performed yet).
        self.loss_history = []

        # Models predict KOs if they are missing one or both KO genes
        self.also_predict_if_missing_ko_gene = True

    def __repr__(self):
        return f"Modelrunner (Model={self.model},\n optimizer={self.optimizer.__class__.__name__},\n optimizer_kwargs={self.optimizer_kwargs},\n nr_epochs={self.nr_epochs},\n batch_size={self.batch_size}\n)"

    def get_save_file_name(self) -> str:
        file_name = f"{self.file_name_prefix}{self.model.name}"
        for hyperparam_name, hyperparam_value in self.model.get_hyperparams():
            if hyperparam_name == "device":
                continue
            file_name += f"_{hyperparam_name}_{hyperparam_value}"
        if self.optimizer_kwargs:
            for kwarg_name, value in self.optimizer_kwargs.items():
                file_name += f"_{kwarg_name}_{value}"
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
    def predict(self, dataset: ExpressionDataWithConditions):
        """Use model to predict the knockouts in the provided dataset

        Args:
            dataset (ExpressionDataWithConditions): Dataset on which to predict.

        Returns:
            Tuple of three tensors: experiment_data (Experiment x (Gene + Nr_conditions)), ground_truth_expressions (Experiment x Gene x Time),
              and model predictions also with dimensions (Experiment x Gene x Time)
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

    def run_for_parallelisation_and_save(self):
        self.train()
        self.save(self.save_path)

        logging.info(f"Finished training and saving model to {self.save_path}")

    def save(self, checkpoint_path: Path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No existing model found at {self.save_path}.")

        logging.info(f"Loading from {self.save_path}")

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

    def load_if_exists(self):
        try:
            self.load(self.save_path)
            return True
        except FileNotFoundError as e:
            logging.warning(e)
            return False

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

    def _tidy_df_of_predictions_on_dataset(
        self,
        dataset: ExpressionDataWithConditions,
    ):
        # TODO move this entire bit ot the TidyDfResults class?
        all_predictions = []
        self.model.eval()
        # Check if this makes sense
        for ko in tqdm(dataset._ko_dict.keys()):
            try:
                condition, ground_truth_expressions = (
                    dataset._get_condition_and_expression_under_ko_gene(
                        ko, handle_missing_ko_genes=self.also_predict_if_missing_ko_gene
                    )
                )
            except KeyError:
                continue
        #         if self.also_predict_if_missing_ko_gene:
        #             condition = list(dataset._ko_dict.keys())[0]
        #             # Just use first condition to infer which initial conditions to provide
        #             ground_truth_expressions = dataset._ko_dict[
        #                 condition
        #             ]

        #             # Check which genes are still in dataset. Proceed with these for predictions. (e.g. only knockout one instead of both genes)
        #             kos_in_dataset = tuple(
        #                 gene for gene in condition.ko_gene if gene in ko
        #             )
        #             condition = ExperimentCondition(
        #                 ko_gene=kos_in_dataset,
        #                 all_genes=condition.all_genes,
        #                 environment=condition.environment,
        #                 all_environments=condition.all_environments,
        #             )

        #         else:
        #             continue

        #     assert (
        #         dataset.genes == self.model.modelled_gene_labels
        #     ), "Gene labels do not match"

        #     initial_state = ground_truth_expressions.as_tensor()[:, 0].unsqueeze(0)
        #     condition_tensor = condition.as_tensor().unsqueeze(0)
        #     initial_state = initial_state.to(self.model.device)
        #     condition_tensor = condition_tensor.to(self.model.device)

        #     with torch.no_grad():
        #         prediction = self._calculate_model_prediction_at_conditions(
        #             condition_tensor, initial_state
        #         )
        #         prediction = prediction.squeeze(0).detach().cpu().numpy()

        #     df_predicted = pd.DataFrame(
        #         prediction,
        #         columns=self.dataset._get_time_points(),
        #         index=self.dataset.genes,
        #     )
        #     df_predicted = df_predicted.melt(
        #         ignore_index=False,
        #         var_name=TidyDfResults.TIME_COL,
        #         value_name=TidyDfResults.EXPRESSION_COL,
        #     ).reset_index(names=TidyDfResults.GENE_COL)

        #     model_title = self.get_save_file_name().removesuffix(self.file_name_suffix)
        #     model_title = model_title.removeprefix(self.file_name_prefix)
        #     df_predicted[TidyDfResults.MODEL_COL] = model_title

        #     if ko == condition.ko_gene:
        #         # Add the ground truth measurements, too.
        #         # Only if the ground truth for the KO was saved in this case.
        #         ground_truth_df = ground_truth_expressions.df.melt(
        #             ignore_index=False,
        #             var_name=TidyDfResults.TIME_COL,
        #             value_name=TidyDfResults.EXPRESSION_COL,
        #         ).reset_index(names=TidyDfResults.GENE_COL)
        #         ground_truth_df[TidyDfResults.MODEL_COL] = (
        #             TidyDfResults.GROUND_TRUTH_COL
        #         )
        #         full_df = pd.concat([ground_truth_df, df_predicted])
        #     elif self.also_predict_if_missing_ko_gene:
        #         full_df = df_predicted
        #     else:
        #         raise ValueError("This should not happen.")

        #     full_df[TidyDfResults.CONDITION_COL] = f"KO: {', '.join(ko)}"
        #     all_predictions.append(full_df)
        # self.model.train()
        # return all_predictions


def run_training_job(job: ModelRunner):
    """Runs one training job. Can be called from mp.Pool to run multiple in parallel"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(message)s",
        force=True,
    )
    try:
        output = job.run_for_parallelisation_and_save()
    except AssertionError as e:
        logging.error(f"Error occurred while running job: {e}")
        output = {"error": str(e)}
    return output
