import torch
from torch import nn
from mlmmint.Models.Model import Model


class NeuralODE(Model):
    def __init__(self, modelled_gene_labels):
        super().__init__(modelled_gene_labels)
        # Create a fully connected neural network
        self.neural_net = nn.Sequential(
            nn.Linear(self.n_genes, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_genes),
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.neural_net(state)


class CellBox(Model):
    def __init__(self, modelled_gene_labels):
        super().__init__(modelled_gene_labels)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pass
