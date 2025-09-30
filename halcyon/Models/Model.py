from abc import ABC, abstractmethod
from typing import List
from natsort import natsorted
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class Model(ABC, nn.Module):
    def __init__(self, modelled_gene_labels: List[str]):
        super().__init__()
        self.modelled_gene_labels = natsorted(modelled_gene_labels)
        self.gene_index = {gene: i for i, gene in enumerate(self.modelled_gene_labels)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def n_genes(self):
        return len(self.modelled_gene_labels)

    @abstractmethod
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Get the derivative at a given time point."""
        pass

    @abstractmethod
    def set_conditions(self, conditions: torch.Tensor) -> None:
        """Set the environmental conditions."""
        pass

    def integrate(
        self,
        initial_state: torch.Tensor,
        t_eval: torch.Tensor,
        stress=None,
    ) -> torch.Tensor:
        """Solve the system over time. Provide initial state in form of [Batch, Genes]. Returns tensor with [Batch, Genes, Time]."""
        output: torch.Tensor = odeint(self, initial_state, t_eval)  # type: ignore
        # Permute dimensions for some reason
        return output.permute(1, 2, 0)
