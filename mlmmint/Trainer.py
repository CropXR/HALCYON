from matplotlib import pyplot as plt
import pandas as pd
import torch
from mlmmint.ExpressionData import ExpressionData
from typing import Callable
from mlmmint.Models.Model import Model
from torch.nn.functional import mse_loss


class Trainer:
    def __init__(
        self,
        model: Model,
        expressions: ExpressionData,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs=None,
        loss_fn: Callable = mse_loss,
    ):
        self.model = model
        self.loss_fn = loss_fn
        # Ensure that expressions match the order of the modelled genes
        self.expressions = expressions.match_genes_to_list(model.modelled_gene_labels)
        if optimizer_kwargs is None:
            optimizer_kwargs = dict()
        self.optimizer = optimizer_fn(model.parameters(), **optimizer_kwargs)
        self.model.to(self.model.device)

    def get_initial_states(self) -> torch.Tensor:
        """Get inital states (i.e. the expression levels at time 0 for the expressiondata)."""
        initial_states = self.expressions._get_initial_states()
        initial_states_tensor = torch.tensor(initial_states, dtype=torch.float32)
        return initial_states_tensor

    def get_time_points(self) -> torch.Tensor:
        """Get time points at which the expressions were measured."""
        time_points = self.expressions._get_time_points()
        time_points_tensor = torch.tensor(time_points, dtype=torch.float32)
        return time_points_tensor

    def train(self, n_iters: int):
        """Train the model for a given number of iterations using autograd."""

        loss_history = []
        for itr in range(n_iters):
            self.optimizer.zero_grad()
            output = self.integrate_model()
            loss = self.calculate_loss(output)
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())
            if itr % 20 == 0:
                print(f"Iteration {itr}: Loss {loss}")
        return loss_history

    def calculate_loss(self, output: torch.Tensor):
        """Calculate the loss between a given model output and the actual expressions."""
        return self.loss_fn(self.expressions.as_tensor(), output)

    def plot_fit_vs_real(self):
        """Plot fitted model vs measured data"""
        fig, axs = plt.subplots(1, 2, sharey=True)
        axs[0].set_title("Measured")
        axs[1].set_title("Model")
        self.expressions.plot_expressions(axis=axs[0])

        model_expression_outputs = self.model_output_as_expression_data_object()
        model_expression_outputs.plot_expressions(axis=axs[1])
        plt.tight_layout()
        return fig, axs

    def model_output_as_expression_data_object(self) -> ExpressionData:
        """Integrate the model and return the output as ExpressionData"""
        model_outputs = self.integrate_model()
        model_output_df = pd.DataFrame(
            model_outputs.detach().numpy(),
            columns=self.expressions._get_time_points(),
            index=self.expressions.genes,
        )
        return ExpressionData(model_output_df)

    def integrate_model(
        self,
        initial_states_tensor: torch.Tensor | None = None,
        time_points_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the model over time

        If no initital states or time points are provided, integrate the model over time starting with same initial conditions as in the
        expression data and evaluating at the same time points as in the expressions data. Otherwise calculate data for user-specified initial conditions and time points.

        Args:
            initial_states_tensor (torch.Tensor | None, optional): If specified, use these custom initial states. Defaults to None.
            time_points_tensor (torch.Tensor | None, optional): If specified, evaluate ODE integration at exactly these time points. Defaults to None.

        Returns:
            torch.Tensor: Tensor with genes as rows and time points as columns.
        """

        if initial_states_tensor is None:
            initial_states_tensor = self.get_initial_states()
        if time_points_tensor is None:
            time_points_tensor = self.get_time_points()

        initial_states_tensor = initial_states_tensor.to(self.model.device)
        time_points_tensor = time_points_tensor.to(self.model.device)

        output = self.model.integrate(
            initial_state=initial_states_tensor, t_eval=time_points_tensor
        )
        return output
