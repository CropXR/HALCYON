import torch
from mlmmint.ExpressionData import ExpressionData
from mlmmint.Models.Model import Model
from torch.nn.functional import mse_loss


class Trainer:
    def __init__(self, model: Model, loss_fn=mse_loss):
        self.loss_fn = loss_fn
        self.model = model

    def prepare_integration_inputs(self, expressions: ExpressionData):
        initial_states = expressions.get_initial_states()
        time_points = expressions.get_time_points()
        initial_states_tensor = torch.Tensor(initial_states)
        time_points_tensor = torch.Tensor(time_points)
        return initial_states_tensor, time_points_tensor

    def train(self, expressions: ExpressionData):
        initial_states_tensor, time_points_tensor = self.prepare_integration_inputs(
            expressions
        )
        output = self.integrate_model(
            self.model, initial_states_tensor, time_points_tensor
        )
        loss = self.calculate_loss(expressions, output)
        # # Update weights
        # self.model.optimizer.zero_grad()
        # loss.backward()

    def calculate_loss(self, expressions: ExpressionData, output: torch.Tensor):
        return self.loss_fn(output, expressions.as_tensor())

    def integrate_model(self, model, initial_states_tensor, time_points_tensor):
        output = model.integrate(
            initial_state=initial_states_tensor, t_eval=time_points_tensor
        )
        return output
