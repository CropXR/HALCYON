import logging
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

    def parse_env_data(self, environment_data) -> tuple[torch.Tensor, torch.Tensor]:
        knockout_flags = environment_data[:, : self.n_genes]
        env_flags = environment_data[:, self.n_genes :]
        # Build batch-specific masks: if gene j is knocked out in batch b,
        # we zero out all incoming connections to gene j
        # i.e., we zero row j of the weight matrix in that batch
        knockout_masks = 1.0 - knockout_flags.unsqueeze(2)  # [B, G, 1]
        return knockout_masks, env_flags


class NeuralODE(MachineLearningModel):
    def __init__(
        self, modelled_gene_labels: list[str], condition_names: tuple[str, ...]
    ):
        super().__init__(modelled_gene_labels)
        self.n_conditions = len(condition_names)
        # Create a fully connected neural network
        self.neural_net = nn.Sequential(
            nn.Linear(self.n_genes + self.n_conditions, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_genes),
        )

    def set_conditions(self, env_data):
        self.reset_conditions()
        assert (
            env_data.shape[1] == self.n_genes + self.n_conditions
        ), "Environment data should be of shape (B, G + C) where B is the batch size, G is the number of genes and C is the number of conditions"
        # Knockouts should be used as mask after the forwards pass
        ko_masks, env_flags = self.parse_env_data(env_data)
        self.env_flags = env_flags
        # Environments should be appended to the forward pass but not returned (Because they do not change over time)
        self.register_buffer("ko_masks", ko_masks)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        input = torch.cat((state, self.env_flags), dim=1)  # type: ignore
        return self.neural_net(input) * self.ko_masks.squeeze(2)  # type: ignore

    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        raise NotImplementedError

    def reset_conditions(self):
        self.register_buffer("ko_masks", torch.ones(self.n_genes))  # type: ignore


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
        self.reset_conditions()

    @property
    def w_masked(self) -> torch.Tensor:
        return self.W * self.w_mask  # type: ignore

    def reset_conditions(self):
        mask_self_interaction = ~torch.eye(self.n_genes, dtype=torch.bool)
        self.register_buffer("w_mask", mask_self_interaction)

    def set_conditions(self, environment_data: torch.Tensor):
        self.reset_conditions()
        knockout_masks, _ = self.parse_env_data(environment_data)
        new_w_masked = self.w_masked.unsqueeze(0) * knockout_masks  # [B, G, G]
        assert not torch.isnan(new_w_masked).any(), "NaN in w_masked"
        self.register_buffer("w_mask", new_w_masked)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Batch matrix multiplication: [B, G, G] @ [B, G, 1] -> [B, G]
        input_term = torch.bmm(
            self.w_masked,
            state.unsqueeze(2),  # type: ignore
        ).squeeze(2)
        # Decay always has to be negative, hence the softplus
        decay = nn.functional.softplus(self.decay_params, beta=10)
        gene_expression_changes = self.epsilon * self.phi(input_term) - decay * state
        return gene_expression_changes

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
        w_masked = self.w_masked.detach().numpy()
        G = nx.DiGraph()
        for i, gene_to in enumerate(self.modelled_gene_labels):
            for j, gene_from in enumerate(self.modelled_gene_labels):
                weight = w_masked[i, j]
                if threshold is None or abs(weight) > threshold:
                    G.add_edge(gene_from, gene_to, influence=weight)
        return RegulatoryNetwork(G)
