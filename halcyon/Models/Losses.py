import torch
import torch.nn as nn

from halcyon.Models.GeneExpressionModel import GeneExpressionModel
from halcyon.Models.HybridModel import HybridModel


class MSELossFlexible(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, expression_pred, expression_true, *args, **kwargs):
        """Ignores extra arguments, because might be passing along the model parameters as potential third argument"""
        return self.mse(expression_pred, expression_true)


class RegularisingMSE(MSELossFlexible):
    def __init__(self, lambda_param: float = 1.0):
        super().__init__()
        self.lambda_param = lambda_param

    def __repr__(self):
        return f"{self.__class__.__name__} with lambda={self.lambda_param}"


class MSELossPlusMachineLearningL2(RegularisingMSE):
    def __init__(self, lambda_param: float):
        super().__init__(lambda_param)

    def forward(
        self,
        expression_pred: torch.Tensor,
        expression_true: torch.Tensor,
        model: HybridModel,
    ):
        # Only take losses from ML model parameters and ignore the per gene multipliers (because it is okay to have these be 1 I think?)
        ml_params = [
            p
            for n, p in model._ml.named_parameters()
            if not n == "per_gene_multipliers"
        ]
        total_params_squared = sum((p**2).sum() for p in ml_params)
        total_params = sum(p.numel() for p in ml_params)
        loss_l2 = total_params_squared / total_params
        return self.mse(expression_pred, expression_true) + self.lambda_param * loss_l2


class MSELossPlusL1(RegularisingMSE):
    def __init__(self, lambda_param: float):
        super().__init__(lambda_param)

    def forward(
        self,
        expression_pred: torch.Tensor,
        expression_true: torch.Tensor,
        model: GeneExpressionModel,
    ):
        mse_loss = self.mse(expression_pred, expression_true)
        total_params_abs = sum(
            p.abs().sum() for name, p in model.named_parameters() if "bias" not in name
        )
        total_params = sum(p.numel() for name, p in model.named_parameters())
        loss_l1 = total_params_abs / total_params
        return mse_loss + self.lambda_param * loss_l1
