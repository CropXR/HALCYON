import logging
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions

from typing import Callable
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

    def train(self, n_epochs: int, do_profiling=False):
        """Perform model training on batches.

        Args:
            n_epochs (int): Number of epochs to train
        """
        loss_history = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in self.dataloader:
                experiment_data, expression_data = batch
                experiment_data = experiment_data.to(self.model.device)
                expression_data = expression_data.to(self.model.device)

                self.optimizer.zero_grad()
                self.model.set_conditions(experiment_data)
                output = self.integrate_model(
                    initial_states_tensor=expression_data[:, :, 0]
                )

                loss = self.calculate_loss(expression_data, output)
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

    def calculate_loss(
        self, expression_true: torch.Tensor, expression_pred: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss between a given model output and the actual expressions."""
        return self.loss_fn(expression_pred, expression_true)

    def plot_fit_vs_real(self, top_n=None, show_legend: bool = True):
        """Plot fitted model vs measured data"""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty.")
        if top_n is not None:
            n = top_n
        else:
            n = len(self.dataset)
        fig, axs = plt.subplots(n, 2, sharey=True, figsize=(10, 4 * n))
        if n == 1:
            axs = np.expand_dims(
                axs, axis=0
            )  # ensure consistent indexing for single row

        timepoints = self.dataset._get_time_points()
        gene_labels = self.dataset.genes
        for i in range(n):
            env, expressions = self.dataset[i]
            env = env.to(self.model.device)
            expressions = expressions.to(self.model.device)
            # Plot ground truth
            df_real = pd.DataFrame(
                expressions.detach().cpu().numpy(),
                columns=timepoints,
                index=gene_labels,
            )
            ExpressionData(df_real).plot_expressions(axis=axs[i, 0], legend=show_legend)
            axs[i, 0].set_title(f"Measured (Sample {i})")

            # Simulate and plot prediction
            self.model.set_conditions(env.unsqueeze(0))
            output = self.integrate_model(
                initial_states_tensor=expressions[:, 0].unsqueeze(0)
            )
            df_pred = pd.DataFrame(
                output.squeeze(0).detach().cpu().numpy(),
                columns=timepoints,
                index=gene_labels,
            )
            ExpressionData(df_pred).plot_expressions(axis=axs[i, 1], legend=show_legend)
            axs[i, 1].set_title(f"Model (Sample {i})")
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
            "loss_history": self.loss_history,  # e.g. a list of floats
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_history = checkpoint["loss_history"]
        self.model.load_state_dict(checkpoint["model_state_dict"])


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
        model = model_cls(
            self.expressions.genes,
            self.expressions.all_environments,
            **self.model_kwargs,
        )
        self.trainer = Trainer(
            model, self.expressions, batch_size=32, optimizer_kwargs=optimizer_kwargs
        )

        file_name = f"trained_trainer_{model.__class__.__name__}"
        if self.model_kwargs:
            for kwarg_name, value in self.model_kwargs.items():
                if kwarg_name == "device":
                    continue
                file_name += f"_{kwarg_name}_{value}"
        file_name += ".pt"

        self.save_path = self.sim_result.data_path / file_name

    def __repr__(self):
        return f"Training {self.trainer.model.__class__.__name__} on {self.sim_result} with {self.model_kwargs}"

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
