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

    def get_initial_states_and_time_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get inital states (i.e. the expression levels at time 0) and time points."""
        initial_states = self.expressions._get_initial_states()
        time_points = self.expressions._get_time_points()
        initial_states_tensor = torch.tensor(initial_states, dtype=torch.float32)
        time_points_tensor = torch.tensor(time_points, dtype=torch.float32)
        return initial_states_tensor, time_points_tensor

    def train(self, n_iters: int):
        """Train the model for a given number of iterations using autograd."""

        loss_history = []
        for itr in range(n_iters):
            self.optimizer.zero_grad()
            output = self.integrate_model_with_init_and_t_from_expressions()
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

    def integrate_model_with_init_and_t_from_expressions(self) -> torch.Tensor:
        """Integrate the model over time starting with same initial conditions as in the expression data and evaluating at the same time points as in the expressions data.

        Returns tensor with genes as rows and time points as columns."""
        initial_states_tensor, time_points_tensor = (
            self.get_initial_states_and_time_points()
        )
        return self.integrate_model_with_custom_init_and_t(
            initial_states_tensor, time_points_tensor
        )

    def plot_fit_vs_real(self):
        fig, axs = plt.subplots(1, 2, sharey=True)
        axs[0].set_title("Measured")
        axs[1].set_title("Model")
        self.expressions.plot_expressions(axis=axs[0])

        model_expression_outputs = self.model_output_as_expression_data_object()
        model_expression_outputs.plot_expressions(axis=axs[1])
        plt.tight_layout()
        return fig, axs

    def model_output_as_expression_data_object(self) -> ExpressionData:
        model_outputs = self.integrate_model_with_init_and_t_from_expressions()
        model_output_df = pd.DataFrame(
            model_outputs.detach().numpy(),
            columns=self.expressions._get_time_points(),
            index=self.expressions.genes,
        )
        return ExpressionData(model_output_df)

    def integrate_model_with_custom_init_and_t(
        self, initial_states_tensor: torch.Tensor, time_points_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Integrate the model over time for a user-specified initial condition and time points.

        Returns tensor with genes as rows and time points as columns."""
        output = self.model.integrate(
            initial_state=initial_states_tensor, t_eval=time_points_tensor
        )
        return output
