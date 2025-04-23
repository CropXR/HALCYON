import logging
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mlmmint.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from typing import Callable
from mlmmint.Datasets.ExpressionData import ExpressionData
from mlmmint.Models.Model import Model
from torch.nn.functional import mse_loss


class Trainer:
    def __init__(
        self,
        model: Model,
        expressions: ExpressionDataWithConditions,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs=None,
        loss_fn: Callable = mse_loss,
        batch_size: int = 1,
    ):
        self.model = model
        self.loss_fn = loss_fn
        # Ensure that expressions match the order of the modelled genes
        self.dataset = expressions
        self.dataset.match_genes_to_list(model.modelled_gene_labels)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        if optimizer_kwargs is None:
            optimizer_kwargs = dict()
        self.optimizer = optimizer_fn(model.parameters(), **optimizer_kwargs)
        self.model.to(self.model.device)

    def get_time_points(self) -> torch.Tensor:
        """Get time points at which the expressions were measured."""
        time_points = self.dataset._get_time_points()
        time_points_tensor = torch.tensor(time_points, dtype=torch.float32)
        return time_points_tensor

    def train(self, n_epochs: int) -> list[float]:
        """Train the model"""
        loss_history = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in self.dataloader:
                experiment_data, expression_data = batch
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
        return loss_history

    def calculate_loss(
        self, expression_true: torch.Tensor, expression_pred: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss between a given model output and the actual expressions."""
        return self.loss_fn(expression_true, expression_pred)

    def plot_fit_vs_real(self):
        """Plot fitted model vs measured data"""
        fig, axs = plt.subplots(1, 2, sharey=True)
        axs[0].set_title("Measured")
        axs[1].set_title("Model")
        self.dataset.plot_expressions(axis=axs[0])

        model_expression_outputs = self.model_output_as_expression_data_object()
        model_expression_outputs.plot_expressions(axis=axs[1])
        plt.tight_layout()
        return fig, axs

    def model_output_as_expression_data_object(self) -> ExpressionData:
        """Integrate the model and return the output as ExpressionData"""
        model_outputs = self.integrate_model(self.get_time_points())
        model_output_df = pd.DataFrame(
            model_outputs.detach().numpy(),
            columns=self.dataset._get_time_points(),
            index=self.dataset.genes,
        )
        return ExpressionData(model_output_df)

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
