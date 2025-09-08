import torch
from torch import nn


class PhoenixSumActivation(nn.Module):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return (y - 0.5) / (1 + torch.abs(y - 0.5))


class PhoenixProdActivation(nn.Module):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.log(((y - 0.5) / (1 + torch.abs(y - 0.5))) + 1)
