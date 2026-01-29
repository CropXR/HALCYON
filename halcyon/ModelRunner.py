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
from halcyon.TidyDfResults import TidyDfResults
from halcyon.Simulators.SimulationResult import SimulationResult
from tqdm import tqdm


class ModelRunner:
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
        """Class to handle running models on simulation resutls. Handles the integration between dataloader and model.

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
        self.save_path = self.sim_result.data_path / file_name

        # Loss history after training. Is empty initially (because no training has been performed yet).
        self.loss_history = []

        # Models predict KOs if they are missing one or both KO genes
        self.also_predict_if_missing_ko_gene = True

    def get_save_file_name(self) -> str:
        file_name = f"{self.file_name_prefix}{self.model.__class__.__name__}"
        for hyperparam_name, hyperparam_value in self.model.get_hyperparams():
            if hyperparam_name == "device":
                continue
            file_name += f"_{hyperparam_name}_{hyperparam_value}"
        if self.optimizer_kwargs:
            for kwarg_name, value in self.optimizer_kwargs.items():
                file_name += f"_{kwarg_name}_{value}"
        file_name += self.file_name_suffix
        return file_name

    def load_if_exists(self):
        try:
            self.load(self.save_path)
        except FileNotFoundError as e:
            logging.info(f"No existing model found to load at self.save_path. \n {e}")
            return False

    def get_time_points(self, dataset: ExpressionDataWithConditions) -> torch.Tensor:
        """Get time points at which the expressions were measured."""
        time_points = dataset._get_time_points()
        time_points_tensor = torch.tensor(time_points, dtype=torch.float32)
        return time_points_tensor

    def _calculate_model_prediction_at_conditions(
        self, experiment_data: torch.Tensor, initial_state: torch.Tensor
    ):
        self.optimizer.zero_grad()
        self.model.set_conditions(experiment_data)
        output = self.integrate_model(initial_states_tensor=initial_state)
        return output

    def _calculate_model_prediction_from_batch(
        self, batch: torch.Tensor, return_true_expression: bool = False
    ):
        """Provide a batch from the dataloader as input and return model prediction

        Args:
            batch (torch.Tensor): Dataloader input data batch
            return_true_expression (bool, optional): If true, also return the ground truth expression data. Can be used to calculate loss. Defaults to False.

        """
        experiment_data, expression_data = self.preprocess_batch(batch)
        output = self._calculate_model_prediction_at_conditions(
            experiment_data, expression_data[:, :, 0]
        )
        if return_true_expression:
            return output, expression_data
        return output

    def _calculate_loss_from_batch(self, batch: torch.Tensor):
        output, true_expresssion = self._calculate_model_prediction_from_batch(
            batch, return_true_expression=True
        )
        return self.loss_fn(output, true_expresssion, self.model)

    def preprocess_batch(self, batch):
        experiment_data, expression_data = batch
        experiment_data = experiment_data.to(self.model.device)
        expression_data = expression_data.to(self.model.device)
        return experiment_data, expression_data

    def train(self, dataset: ExpressionDataWithConditions):
        """Perform model training."""

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

        loss_history = []

        for epoch in range(self.nr_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                loss = self._calculate_loss_from_batch(batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            mean_epoch_loss = epoch_loss / len(dataloader)
            loss_history.append(mean_epoch_loss)
            logging.info(f"Epoch {epoch}, loss {mean_epoch_loss}")

        self.loss_history = loss_history

    def run_for_parallelisation_and_save(self):
        self.train()
        self.save(self.save_path)

        logging.info(
            f"Finished fitting for genes: {self.sim_result.nr_genes}, modules: {self.sim_result.nr_modules}, sparsity: {self.sim_result.sparsity}, final loss: {self.final_loss}"
        )

    def plot_loss_history(self, ax=None):
        """After training the model, plot the loss history"""
        assert (
            len(self.loss_history) > 0
        ), "No loss history available. Train the model first."
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        sns.lineplot(self.loss_history, ax=ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_ylim((0, 1.5 * max(self.loss_history)))
        return fig, ax

    def integrate_model(
        self,
        initial_states_tensor: torch.Tensor,
        time_points_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the model over time

        If no initital states or time points are provided, integrate the model over time starting with same initial conditions as in the
        expression data and evaluating at the same time points as in the expressions data. Otherwise calculate data for user-specified initial conditions and time points.

        Args:
            time_points_tensor (torch.Tensor | None, optional): If specified, evaluate ODE integration at exactly these time points. Defaults to None.

        Returns:
            torch.Tensor: Tensor with genes as rows and time points as columns.
        """
        if time_points_tensor is None:
            time_points_tensor = self.get_time_points()

        initial_states_tensor = initial_states_tensor.to(self.model.device)
        time_points_tensor = time_points_tensor.to(self.model.device)

        output = self.model.integrate(
            initial_state=initial_states_tensor, t_eval=time_points_tensor
        )
        return output

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

    def _tidy_df_of_predictions_on_dataset(
        self,
        expressions_to_predict: ExpressionDataWithConditions,
        kos_to_predict: list[tuple[str]],
    ):
        all_predictions = []
        self.model.eval()
        for ko in tqdm(kos_to_predict):
            try:
                condition, ground_truth_expressions = (
                    expressions_to_predict._get_condition_and_expression_under_ko_gene(
                        ko, handle_missing_ko_genes=self.also_predict_if_missing_ko_gene
                    )
                )
            except KeyError:
                if self.also_predict_if_missing_ko_gene:
                    condition = list(expressions_to_predict._ko_dict.keys())[0]
                    # Just use first condition to infer which initial conditions to provide
                    ground_truth_expressions = expressions_to_predict._ko_dict[
                        condition
                    ]

                    # Check which genes are still in dataset. Proceed with these for predictions. (e.g. only knockout one instead of both genes)
                    kos_in_dataset = tuple(
                        gene for gene in condition.ko_gene if gene in ko
                    )
                    condition = ExperimentCondition(
                        ko_gene=kos_in_dataset,
                        all_genes=condition.all_genes,
                        environment=condition.environment,
                        all_environments=condition.all_environments,
                    )

                else:
                    continue

            assert (
                expressions_to_predict.genes == self.model.modelled_gene_labels
            ), "Gene labels do not match"

            initial_state = ground_truth_expressions.as_tensor()[:, 0].unsqueeze(0)
            condition_tensor = condition.as_tensor().unsqueeze(0)
            initial_state = initial_state.to(self.model.device)
            condition_tensor = condition_tensor.to(self.model.device)

            with torch.no_grad():
                prediction = self._calculate_model_prediction_at_conditions(
                    condition_tensor, initial_state
                )
                prediction = prediction.squeeze(0).detach().cpu().numpy()

            df_predicted = pd.DataFrame(
                prediction,
                columns=self.dataset._get_time_points(),
                index=self.dataset.genes,
            )
            df_predicted = df_predicted.melt(
                ignore_index=False,
                var_name=TidyDfResults.TIME_COL,
                value_name=TidyDfResults.EXPRESSION_COL,
            ).reset_index(names=TidyDfResults.GENE_COL)

            model_title = self.get_save_file_name().removesuffix(self.file_name_suffix)
            model_title = model_title.removeprefix(self.file_name_prefix)
            df_predicted[TidyDfResults.MODEL_COL] = model_title

            if ko == condition.ko_gene:
                # Add the ground truth measurements, too.
                # Only if the ground truth for the KO was saved in this case.
                ground_truth_df = ground_truth_expressions.df.melt(
                    ignore_index=False,
                    var_name=TidyDfResults.TIME_COL,
                    value_name=TidyDfResults.EXPRESSION_COL,
                ).reset_index(names=TidyDfResults.GENE_COL)
                ground_truth_df[TidyDfResults.MODEL_COL] = (
                    TidyDfResults.GROUND_TRUTH_COL
                )
                full_df = pd.concat([ground_truth_df, df_predicted])
            elif self.also_predict_if_missing_ko_gene:
                full_df = df_predicted
            else:
                raise ValueError("This should not happen.")

            full_df[TidyDfResults.CONDITION_COL] = f"KO: {', '.join(ko)}"
            all_predictions.append(full_df)
        self.model.train()
        return all_predictions

    def predict_dataframe_from_expression_data(
        self, expressions_to_predict: ExpressionDataWithConditions
    ):
        assert (
            expressions_to_predict.genes == self.model.modelled_gene_labels
        ), "Gene labels do not match"
        self.model.eval()

        all_predictions = []
        with torch.no_grad():
            for batch in expressions_to_predict:
                predictions = self._calculate_model_prediction_from_batch(batch)
                all_predictions.append(predictions.detach().cpu().item())
        self.model.train()
        return all_predictions

    def predict_and_calculate_loss_from_expression_data(
        self, expressions_to_predict: ExpressionDataWithConditions
    ):
        """Provide Expressiondata (e.g. with double knockouts) and calculate loss on them.

        Args:
            expressions_to_predict (ExpressionDataWithConditions): Dataset on which losses should be calculated
        """
        assert (
            expressions_to_predict.genes == self.model.modelled_gene_labels
        ), "Gene labels do not match"
        self.model.eval()

        my_dataloader = DataLoader(
            expressions_to_predict, batch_size=1, shuffle=False, pin_memory=True
        )
        all_losses = []
        with torch.no_grad():
            for batch in my_dataloader:
                loss = self._calculate_loss_from_batch(batch)
                all_losses.append(loss.detach().cpu().item())
        self.model.train()
        return all_losses


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
