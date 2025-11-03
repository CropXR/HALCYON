import torch
import torch.nn as nn

from halcyon.Models.HybridModel import HybridModel


class MSELossFlexible(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, expression_pred, expression_true, *args, **kwargs):
        """Ignores extra arguments"""
        return self.mse(expression_pred, expression_true)


class MSELossPlusMachineLearningL2(MSELossFlexible):
    def __init__(self, lambda_param: float = 1.0):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(
        self,
        expression_pred: torch.Tensor,
        expression_true: torch.Tensor,
        model: HybridModel,
    ):
        # Only take losses from ML model parameters and ignore the per gene multipliers (because it is okay to have these be 1 I think?)
        ml_params = [
            p for n, p in model.ml.named_parameters() if not n == "per_gene_multipliers"
        ]
        total_params_squared = sum((p**2).sum() for p in ml_params)
        total_params = sum(p.numel() for p in ml_params)
        param_l2 = total_params_squared / total_params
        return self.mse(expression_pred, expression_true) + self.lambda_param * param_l2
