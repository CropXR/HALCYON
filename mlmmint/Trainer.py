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
        loss_fn: Callable = mse_loss,
    ):
        self.model = model
        self.loss_fn = loss_fn
        # Ensure that expressions match the order of the modelled genes
        self.expressions = expressions.match_genes_to_list(model.modelled_gene_labels)
        self.optimizer = optimizer_fn(model.parameters())

    def get_initial_states_and_time_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get inital states (i.e. the expression levels at time 0) and time points."""
        initial_states = self.expressions._get_initial_states()
        time_points = self.expressions._get_time_points()
        initial_states_tensor = torch.tensor(initial_states, dtype=torch.float32)
        time_points_tensor = torch.tensor(time_points, dtype=torch.float32)
        return initial_states_tensor, time_points_tensor

    def train(self, n_iters: int):
        """Train the model for a given number of iterations using autograd."""
        initial_states_tensor, time_points_tensor = (
            self.get_initial_states_and_time_points()
        )
        loss_history = []
        for itr in range(n_iters):
            self.optimizer.zero_grad()
            output = self.integrate_model(initial_states_tensor, time_points_tensor)
            loss = self.calculate_loss(output)
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())
            if itr % 100 == 0:
                print(f"Iteration {itr}: Loss {loss}")
        return loss_history

    def calculate_loss(self, output: torch.Tensor):
        """Calculate the loss between a given model output and the actual expressions."""
        return self.loss_fn(self.expressions.as_tensor(), output)

    def integrate_model(
        self, initial_states_tensor: torch.Tensor, time_points_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Integrate the model over time. Returns tensor with genes as rows and time points as columns."""
        output = self.model.integrate(
            initial_state=initial_states_tensor, t_eval=time_points_tensor
        )
        return output
