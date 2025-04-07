import torch
from torch import nn
from mlmmint.Models.Model import Model
from abc import abstractmethod
import networkx as nx

from mlmmint.RegulatoryNetwork import RegulatoryNetwork


class MachineLearningModel(Model):
    def __init__(self, modelled_gene_labels):
        super().__init__(modelled_gene_labels)

    @abstractmethod
    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        """Extract the inferred network"""
        pass


class NeuralODE(MachineLearningModel):
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

    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        raise NotImplementedError


class CellBox(MachineLearningModel):
    def __init__(self, modelled_gene_labels, phi=torch.tanh):
        super().__init__(modelled_gene_labels)
        self.W = nn.Parameter(
            torch.zeros((self.n_genes, self.n_genes), dtype=torch.float32)
        )
        self.decay_params = nn.Parameter(torch.zeros(self.n_genes))
        self.epsilon = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.phi = phi
        # Ensure no self-interactions (stored as non-parameters (buffer) because
        # we'll always be masking these and doesnt need to be optimised)
        self.register_buffer(
            "mask_self_interaction", ~torch.eye(self.n_genes, dtype=torch.bool)
        )
        self.register_buffer(
            "mask_knockout_interaction", torch.ones_like(self.W, dtype=torch.bool)
        )

    def knockout_interaction(self, from_gene: str, to_gene: str):
        """Knock out the directed interaction from gene <from_gene> to gene <to_gene>"""
        assert from_gene in self.modelled_gene_labels
        assert to_gene in self.modelled_gene_labels
        new_mask = self.mask_knockout_interaction.clone()  # type: ignore
        new_mask[self.gene_index[to_gene], self.gene_index[from_gene]] = (
            False  # Knockout the interaction
        )
        self.register_buffer("mask_knockout_interaction", new_mask)

    def get_w_masked(self) -> torch.Tensor:
        """Get the weight matrix, but mask e.g. the self interactions and knockouts (if they are specified)"""
        w_masked = self.W * self.mask_self_interaction * self.mask_knockout_interaction  # type: ignore
        return w_masked

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        w_masked = self.get_w_masked()
        # Decay always has to be negative, hence the softplus
        return (
            self.epsilon * self.phi(w_masked @ state)
            - nn.functional.softplus(self.decay_params, beta=10) * state
        )

    def _set_weight_matrix_to_ones(self):
        """For unit testing mainly"""
        self.W = nn.Parameter(torch.ones_like(self.W))

    def _set_decay_matrix_to_ones(self):
        """For unit testing mainly"""
        self.decay_params = nn.Parameter(torch.ones_like(self.decay_params))

    def _set_epsilon_to_zero(self):
        """For unit testing mainly. Do not fully initialise at 0 because then the gradient can disappear"""
        self.epsilon = nn.Parameter(torch.tensor(1e-6, dtype=torch.float32))

    def get_inferred_network(self, threshold: float | None = None) -> RegulatoryNetwork:
        """Return the inferred network

        Args:
            threshold (float, optional): Cutoff to determine from what interaction strength we want
            to predict an edge. Defaults to None. If set to a float, return only edges with weights that have an absolute value larger than this cutoff.

        Returns:
            RegulatoryNetwork
        """
        w_masked = self.get_w_masked().detach().numpy()
        G = nx.DiGraph()
        for i, gene_to in enumerate(self.modelled_gene_labels):
            for j, gene_from in enumerate(self.modelled_gene_labels):
                weight = w_masked[i, j]
                if threshold is None or abs(weight) > threshold:
                    G.add_edge(gene_from, gene_to, influence=weight)
        return RegulatoryNetwork(G)
