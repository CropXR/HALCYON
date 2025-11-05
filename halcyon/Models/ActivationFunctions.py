import torch
from torch import nn


class PhoenixSumActivation(nn.Module):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        shifted = y - 0.5
        return shifted / (1 + torch.abs(shifted))


class PhoenixProdActivation(nn.Module):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        shifted = y - 0.5
        # Log1p implicitly computes log(1 + x)
        return torch.log1p((shifted / (1 + torch.abs(shifted))))
