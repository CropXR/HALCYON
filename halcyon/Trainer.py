import logging
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions

from typing import Callable
from halcyon.Datasets.ExpressionData import ExpressionData
from halcyon.Losses import MSELossFlexible
from halcyon.Models.HybridModel import HybridModel
from halcyon.Models.Model import Model
import torch.nn as nn
import seaborn as sns
from halcyon.TidyDfResults import TidyDfResults
from halcyon.Simulators.GNWSimRunner import SimulationResult
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: Model,
        sim_result: SimulationResult,
        do_time_scaling: bool = True,
        nr_training_steps: int = 10,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs=None,
        loss: nn.Module = MSELossFlexible(),
        batch_size: int = 1,
        file_name_prefix="trained_trainer_",
    ):
        """Class to handle training of models

        Args:
            model (Model): Model to be trained
            sim_result (SimulationResult): Simulator output that the model should be trained on.
            dataset (ExpressionDataWithConditions | None, optional): Dataset to use for training. If None, will use no-noise KO timeseries from sim_result. Defaults to None.
            nr_training_steps (int): How many epochs to run the training
            optimizer_fn (Callable, optional): Optimizer function. Defaults to torch.optim.Adam.
            optimizer_kwargs (dict | None, optional): Kwargs to pass to optimiser (e.g. learning rate). Defaults to None.
            loss (Callable, optional): Loss functions. Defaults to mse loss.
            batch_size (int, optional): Batch size to use during training. Defaults to 1.
        """
        self.model = model
        self.loss_fn = loss
        self.nr_training_steps = nr_training_steps

        self.sim_result = sim_result

        self.dataset = self.sim_result.get_no_noise_ko_timeseries_expressiondata()
        self.double_ko_dataset = self.sim_result.get_double_ko_expressiondata()
        self.do_time_scaling = do_time_scaling
        self.file_name_prefix = file_name_prefix
        self.file_name_suffix = ".pt"

        for my_dataset in [self.dataset, self.double_ko_dataset]:
            # Normalise time scale to be from 0 to 1. Helps ODEs not explode
            if self.do_time_scaling:
                scaling_factors = []
                original_max = max(my_dataset._get_time_points())
                my_dataset.scale_time_series()
                new_max = max(my_dataset._get_time_points())
                scaling_factor = torch.tensor(original_max / new_max)
                scaling_factors.append(scaling_factor)

            # Ensure that expressions match the order of the modelled genes
            my_dataset.match_genes_to_list(model.modelled_gene_labels)

        if self.do_time_scaling:
            assert all(
                x == scaling_factors[0] for x in scaling_factors
            ), "Not all timescalers are identical. Check if same timescale used by both double and single KO"
            self.model.set_timescaler(scaling_factor)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs

        self.optimizer = optimizer_fn(model.parameters(), **self.optimizer_kwargs)
        self.model.to(self.model.device)

        file_name = self.get_save_file_name()
        self.save_path = self.sim_result.data_path / file_name

        # Loss history after training. Is empty initially (because no training has been performed yet).
        self.loss_history = []

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

    def __repr__(self):
        return f"Training {self.model.get_name()} on {self.sim_result}. Saving to {self.save_path}"

    def load_if_exists(self):
        try:
            self.load(self.save_path)
        except FileNotFoundError as e:
            logging.info(f"No existing model found to load at self.save_path. \n {e}")
            return False

    @property
    def final_loss(self):
        return self.loss_history[-1] if self.loss_history else np.nan

    def get_time_points(self) -> torch.Tensor:
        """Get time points at which the expressions were measured."""
        time_points = self.dataset._get_time_points()
        time_points_tensor = torch.tensor(time_points, dtype=torch.float32)
        return time_points_tensor

    def summary(self) -> dict:
        output_dict = {}
        output_dict["final_loss"] = self.final_loss
        output_dict["nr_training_steps"] = self.nr_training_steps
        output_dict["model_path"] = str(self.save_path)
        output_dict["model_cls"] = self.model.__class__.__name__
        for key, value in self.sim_result.items():
            output_dict[key] = value
        for key, value in self.model.get_hyperparams():
            output_dict[key] = value
        for key, value in self.optimizer_kwargs.items():
            output_dict[key] = value
        return output_dict

    def losses_on_double_ko(self):
        double_ko_expressions = self.sim_result.get_double_ko_expressiondata()
        loss = self.predict_and_calculate_loss_from_expression_data(
            double_ko_expressions
        )
        return loss

    def _plot_one_prediction(
        self,
        initial_condition: torch.Tensor,
        title: str,
        env: torch.Tensor,
        axs: Axes,
        show_legend: bool,
        **kwargs,
    ):
        """Simulate and plot prediction"""
        with torch.no_grad():
            predictions = self._calculate_model_prediction_at_conditions(
                env.unsqueeze(0), initial_condition
            )
            predictions = predictions.squeeze(0).detach().cpu().numpy()
            self._plot_from_expressions(predictions, title, axs, show_legend, **kwargs)

    def _calculate_model_prediction_at_conditions(
        self, experiment_data: torch.Tensor, initial_state: torch.Tensor
    ):
        self.optimizer.zero_grad()
        self.model.set_conditions(experiment_data)
        output = self.integrate_model(initial_states_tensor=initial_state)
        return output

    def _calculate_model_prediction_from_batch(
        self, batch: torch.Tensor, return_true_expression=False
    ):
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

    def train(self):
        """Perform model training on batches."""
        loss_history = []

        if self.nr_training_steps == 0:
            # Cannot train mechanistic model so just calculate loss on dataset as only loss data point.
            loss_history.append(
                np.mean(
                    self.predict_and_calculate_loss_from_expression_data(self.dataset)
                )
            )

        for epoch in range(self.nr_training_steps):
            epoch_loss = 0.0
            for batch in self.dataloader:
                loss = self._calculate_loss_from_batch(batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            mean_epoch_loss = epoch_loss / len(self.dataloader)
            loss_history.append(mean_epoch_loss)
            logging.info(f"Epoch {epoch}, loss {mean_epoch_loss}")

        self.loss_history = loss_history

    def run_for_parallelisation_and_save(self):
        self.train()
        self.save(self.save_path)

        logging.info(
            f"Finished fitting for genes: {self.sim_result.nr_genes}, modules: {self.sim_result.nr_modules}, sparsity: {self.sim_result.sparsity}, final loss: {self.final_loss}"
        )
        return self.summary()

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

    def _plot_from_expressions(
        self,
        expressions: np.ndarray,
        title: str | None,
        axs: Axes,
        show_legend: bool,
        **kwargs,
    ):
        df_real = pd.DataFrame(
            expressions,
            columns=self.dataset._get_time_points(),
            index=self.dataset.genes,
        )
        ExpressionData(df_real).plot_expressions(axis=axs, legend=show_legend, **kwargs)
        if title is not None:
            axs.set_title(title)

    def plot_fit_vs_real(self, top_n=None, show_legend: bool = True):
        """Plot fitted model vs measured data"""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty.")
        if top_n is not None:
            n = top_n
        else:
            n = len(self.dataset)
        fig, axs = plt.subplots(n, 2, sharey=True, sharex="col", figsize=(7, 3 * n))
        if n == 1:
            axs = np.expand_dims(
                axs, axis=0
            )  # ensure consistent indexing for single row

        for i in range(n):
            env, expressions = self.dataset[i]
            env = env.to(self.model.device)
            expressions = expressions.to(self.model.device)
            # Plot ground truth
            self._plot_from_expressions(
                expressions.detach().cpu().numpy(),
                f"Measured (Sample {i})",
                axs[i, 0],
                show_legend,
            )
            self._plot_one_prediction(
                initial_condition=expressions[:, 0].unsqueeze(0),
                title=f"Model fit (Sample {i})",
                env=env,
                axs=axs[i, 1],
                show_legend=show_legend,
            )

        plt.tight_layout()
        return fig, axs

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
                        ko
                    )
                )
            except KeyError:
                # Gene(s) not in model. Just do not return a dataframe for these?
                continue
            assert (
                expressions_to_predict.genes == self.model.modelled_gene_labels
            ), "Gene labels do not match"

            if self.do_time_scaling:
                expressions_to_predict.scale_time_series()

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
            # Add the ground truth measurements, too.
            ground_truth_df = ground_truth_expressions.df.melt(
                ignore_index=False,
                var_name=TidyDfResults.TIME_COL,
                value_name=TidyDfResults.EXPRESSION_COL,
            ).reset_index(names=TidyDfResults.GENE_COL)
            ground_truth_df[TidyDfResults.MODEL_COL] = TidyDfResults.GROUND_TRUTH_COL

            full_df = pd.concat([ground_truth_df, df_predicted])
            full_df[TidyDfResults.CONDITION_COL] = f"KO: {', '.join(condition.ko_gene)}"
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
        if self.do_time_scaling:
            expressions_to_predict.scale_time_series()

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
        if self.do_time_scaling:
            expressions_to_predict.scale_time_series()
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


def run_training_job(job: Trainer):
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
