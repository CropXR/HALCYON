from collections import UserList
import logging
from pathlib import Path
from typing_extensions import Self
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions

from typing import Callable, Generic, TypeVar
from halcyon.Datasets.ExpressionData import ExpressionData
from halcyon.Models.Model import Model
from torch.nn.functional import mse_loss
import seaborn as sns

from halcyon.Simulators.GNWSimRunner import SimulationResult


class Trainer:
    def __init__(
        self,
        model: Model,
        dataset: ExpressionDataWithConditions,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs=None,
        loss_fn: Callable = mse_loss,
        batch_size: int = 1,
    ):
        """Class to handle training of models

        Args:
            model (Model): Model to be trained
            dataset (ExpressionDataWithConditions): Dataset that contains expressions and their associated conditions.
            optimizer_fn (Callable, optional): Optimizer function. Defaults to torch.optim.Adam.
            optimizer_kwargs (_type_, optional): Optimizer kwargs. Defaults to None.
            loss_fn (Callable, optional): Loss functions. Defaults to mse_loss.
            batch_size (int, optional): Batch size to use during training. Defaults to 1.
        """
        self.model = model
        self.loss_fn = loss_fn
        # Ensure that expressions match the order of the modelled genes
        self.dataset = dataset
        self.dataset.match_genes_to_list(model.modelled_gene_labels)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        if optimizer_kwargs is None:
            optimizer_kwargs = dict()
        self.optimizer = optimizer_fn(model.parameters(), **optimizer_kwargs)
        self.model.to(self.model.device)
        # Loss history after training. Is empty initially (because no training has been performed yet).
        self.loss_history = []

    def get_time_points(self) -> torch.Tensor:
        """Get time points at which the expressions were measured."""
        time_points = self.dataset._get_time_points()
        time_points_tensor = torch.tensor(time_points, dtype=torch.float32)
        return time_points_tensor

    def _calculate_loss_from_batch(self, batch):
        experiment_data, expression_data = batch
        experiment_data = experiment_data.to(self.model.device)
        expression_data = expression_data.to(self.model.device)

        self.optimizer.zero_grad()
        self.model.set_conditions(experiment_data)
        output = self.integrate_model(initial_states_tensor=expression_data[:, :, 0])

        loss = self.calculate_loss_expression_vs_pred(expression_data, output)
        return loss

    def train(self, n_epochs: int):
        """Perform model training on batches.

        Args:
            n_epochs (int): Number of epochs to train
        """
        loss_history = []

        for epoch in range(n_epochs):
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

    def plot_loss_history(self):
        """After training the model, plot the loss history"""
        assert (
            len(self.loss_history) > 0
        ), "No loss history available. Train the model first."
        sns.lineplot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim((0, 1.5 * max(self.loss_history)))
        plt.show()

    def calculate_loss_expression_vs_pred(
        self, expression_true: torch.Tensor, expression_pred: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss between a given model output and the actual expressions."""
        return self.loss_fn(expression_pred, expression_true)

    def _from_expressions_to_plot(
        self, expressions, title: str | None, axs, show_legend
    ):
        df_real = pd.DataFrame(
            expressions,
            columns=self.dataset._get_time_points(),
            index=self.dataset.genes,
        )
        ExpressionData(df_real).plot_expressions(axis=axs, legend=show_legend)
        if title is not None:
            axs.set_title(title)

    def _plot_one_prediction(self, initial_condition, title, env, axs, show_legend):
        # Simulate and plot prediction
        self.model.set_conditions(env.unsqueeze(0))
        predictions = self.integrate_model(initial_states_tensor=initial_condition)
        predictions = predictions.squeeze(0).detach().cpu().numpy()
        self._from_expressions_to_plot(predictions, title, axs, show_legend)

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
            self._from_expressions_to_plot(
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
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_history = checkpoint["loss_history"]
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def verify_on_double_ko(
        self, double_ko_expression_data: ExpressionDataWithConditions
    ):
        assert (
            double_ko_expression_data.genes == self.model.modelled_gene_labels
        ), "Gene labels do not match"
        self.model.eval()
        double_ko_expression_data.scale_time_series()
        my_dataloader = DataLoader(
            double_ko_expression_data, batch_size=1, shuffle=False, pin_memory=True
        )
        all_losses = []
        for batch in my_dataloader:
            loss = self._calculate_loss_from_batch(batch)
            all_losses.append(loss.detach().cpu().item())
        self.model.train()
        # TODO plot predictions if we want
        return all_losses


class TrainingJob:
    def __init__(
        self,
        model_cls: Model,
        sim_result: SimulationResult,
        nr_training_steps: int,
        model_kwargs: dict | None = None,
        optimizer_kwargs: dict | None = None,
    ):
        """Training job that can be used to run training in parallel.

        Args:
            model_cls (Model): Class that can be used to construct the model. Should inherit from Model
            sim_result (SimulationResult): Simulator output that the model should be trained on.
            nr_training_steps (int): How many epochs to run the training
            model_kwargs (dict, optional): Kwargs to pass to model (e.g. device). Defaults to None.
            optimizer_kwargs (dict | None, optional): Kwargs to pass to optimiser (e.g. learning rate). Defaults to None.
        """
        self.sim_result = sim_result
        self.nr_training_steps = nr_training_steps
        self.expressions = (
            ExpressionDataWithConditions.from_genenetweaver_ko_timeseries_output(
                self.sim_result.data_path
                / f"{self.sim_result.network_name}_nonoise_knockout_timeseries.tsv"
            )
        )
        self.expressions.scale_time_series()
        self.model_kwargs = model_kwargs or {}
        self.optimizer_kwargs = optimizer_kwargs or {}
        model = model_cls(
            self.expressions.genes,
            self.expressions.all_environments,
            **self.model_kwargs,
        )
        self.trainer = Trainer(
            model,
            self.expressions,
            batch_size=32,
            optimizer_kwargs=self.optimizer_kwargs,
        )

        file_name = f"trained_trainer_{model.__class__.__name__}"
        if self.model_kwargs:
            for kwarg_name, value in self.model_kwargs.items():
                if kwarg_name == "device":
                    continue
                file_name += f"_{kwarg_name}_{value}"
        if self.optimizer_kwargs:
            for kwarg_name, value in self.optimizer_kwargs.items():
                file_name += f"_{kwarg_name}_{value}"
        file_name += ".pt"

        self.save_path = self.sim_result.data_path / file_name

    def __repr__(self):
        return f"Training {self.trainer.model.__class__.__name__} on {self.sim_result} with {self.model_kwargs} and {self.optimizer_kwargs}"

    def load_if_exists(self):
        if self.save_path.exists():
            logging.info(f"Loading from {self.save_path}")
            self.trainer.load(self.save_path)
        else:
            raise FileNotFoundError(f"No existing model found at {self.save_path}.")

    def run(self):
        self.trainer.train(self.nr_training_steps)
        self.trainer.save(self.save_path)

        logging.info(
            f"Finished fitting for genes: {self.sim_result.nr_genes}, modules: {self.sim_result.nr_modules}, sparsity: {self.sim_result.sparsity}, final loss: {self.final_loss}"
        )
        return self.summary()

    @property
    def final_loss(self):
        return self.trainer.loss_history[-1]

    def summary(self) -> dict:
        output_dict = {}
        output_dict["final_loss"] = self.final_loss
        output_dict["nr_training_steps"] = self.nr_training_steps
        output_dict["model_path"] = str(self.save_path)
        output_dict["model_cls"] = self.trainer.model.__class__.__name__
        for key, value in self.sim_result.items():
            output_dict[key] = value
        if self.model_kwargs:
            for key, value in self.model_kwargs.items():
                output_dict[key] = value
        if self.optimizer_kwargs:
            for key, value in self.optimizer_kwargs.items():
                output_dict[key] = value
        return output_dict


def run_training_job(job: TrainingJob):
    """Runs one training job. Can be called from mp.Pool to run multiple in parallel"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(message)s",
        force=True,
    )
    try:
        output = job.run()
    except AssertionError as e:
        logging.error(f"Error occurred while running job: {e}")
        output = {"error": str(e)}
    return output


T = TypeVar("T", bound="TrainingJob")


class ResultList(UserList[T], Generic[T]):
    @classmethod
    def from_train_jobs(cls, train_jobs: list[TrainingJob]) -> Self:
        results = []
        for job in train_jobs:
            try:
                job.load_if_exists()
                results.append(job)
            except FileNotFoundError as e:
                logging.error(f"Error loading model for job {job.save_path}: {e}")
        return cls(results)

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([i.summary() for i in self.data])

    def __getitem__(self, index):
        """Copy numpy-like indexing, where you can select mulitple items via a list or array of indices."""
        if isinstance(index, list) or isinstance(index, np.ndarray):
            return ResultList([self.data[i] for i in index])
        return super().__getitem__(index)

    def plot_all_model_fits_into_one_fig(
        self,
        top_n_genes: int | None = None,
        model_subset_indices: list[int] | None = None,
        model_title_func: Callable[
            [pd.Series], str
        ] = lambda summary: f"{summary['model_cls']}_hidden_{summary.get('hidden_size', 0)}",
    ):
        model_list = (
            self[model_subset_indices] if model_subset_indices is not None else self
        )
        n_cols = len(model_list) + 1
        n_rows = top_n_genes if top_n_genes else len(model_list[0].trainer.dataset)
        fig, axs = plt.subplots(
            n_rows, n_cols, sharey=True, sharex="col", figsize=(4 * n_cols, 3 * n_rows)
        )
        for i, train_job in enumerate(model_list):
            # print(train_job.summary()["model_path"])
            trainer = train_job.trainer

            for j in range(n_rows):
                env, expressions = trainer.dataset[j]
                env = env.to(trainer.model.device)
                expressions = expressions.to(trainer.model.device)
                if i == 0:
                    # Plot ground truth in first column
                    model_title = "Ground truth"
                    trainer._from_expressions_to_plot(
                        expressions.detach().cpu().numpy(),
                        f"{model_title} (KO {j})",
                        axs[j, 0],
                        show_legend=False,
                    )
                if j == 0:
                    # First row should have titles
                    model_title = model_title_func(train_job.summary())
                else:
                    model_title = None
                trainer._plot_one_prediction(
                    expressions[:, 0].unsqueeze(0),
                    model_title,
                    env,
                    axs[j, i + 1],
                    show_legend=False,
                )
        plt.show()
