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

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        W_masked = self.W * self.mask_self_interaction  # type: ignore
        # Decay always has to be negative, hence the softplus
        return (
            self.epsilon * self.phi(W_masked @ state)
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
