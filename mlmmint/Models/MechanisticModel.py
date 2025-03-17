import torch
from mlmmint.RegulatoryNetwork import RegulatoryNetwork
from mlmmint.Models.Model import Model

from torchdiffeq import odeint_adjoint as odeint
from torch import nn


class MechanisticModel(Model):
    def __init__(self, network: RegulatoryNetwork):
        super().__init__(network.node_names)

    def forward(self):
        raise NotImplementedError

    def integrate(self):
        raise NotImplementedError


class TorchLinearODE(MechanisticModel):
    def __init__(self, regulatory_network: RegulatoryNetwork):
        super().__init__(regulatory_network)

        # Initialize the adjacency/weight matrix
        self.weight_matrix = torch.zeros(
            (self.n_genes, self.n_genes), dtype=torch.float32
        )
        self.gene_index = {gene: i for i, gene in enumerate(self.modelled_gene_labels)}

        for (
            from_gene,
            to_gene,
            property_dict,
        ) in regulatory_network.get_all_interactions():
            if from_gene in self.gene_index and to_gene in self.gene_index:
                from_idx = self.gene_index[from_gene]
                to_idx = self.gene_index[to_gene]
                self.weight_matrix[to_idx, from_idx] = property_dict["influence"]
        self.weight_matrix = nn.Parameter(self.weight_matrix, requires_grad=True)

    def forward(self, t, state):
        return self.weight_matrix @ state

    def integrate(self, initial_state, t_eval, stress=None, gene_ko=None):
        output: torch.Tensor = odeint(self, initial_state, t_eval)  # type: ignore
        # Transposed for some reason
        return output.T

    def _make_weight_matrix_zeroes(self):
        """Initialize the weight matrix by setting everything to 0."""
        self.weight_matrix = nn.Parameter(
            torch.zeros_like(self.weight_matrix, dtype=torch.float32),
            requires_grad=True,
        )
