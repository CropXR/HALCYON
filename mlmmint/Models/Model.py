from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn


class Model(ABC, nn.Module):
    def __init__(self, modelled_gene_labels: List[str]):
        super().__init__()
        self.modelled_gene_labels = sorted(modelled_gene_labels)
        self.gene_index = {gene: i for i, gene in enumerate(self.modelled_gene_labels)}

    @property
    def n_genes(self):
        return len(self.modelled_gene_labels)

    @abstractmethod
    def integrate(
        self,
        initial_state: torch.Tensor,
        t_eval: torch.Tensor,
        stress=None,
        gene_ko=None,
    ) -> torch.Tensor:
        """"""
        pass

    @abstractmethod
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Get the derivative at a given time point."""
        pass
