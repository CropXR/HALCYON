from halcyon.Models.MachineLearningModel import MachineLearningModel
from halcyon.RegulatoryNetwork import RegulatoryNetwork


import networkx as nx
import torch
from torch import nn


class CellBox(MachineLearningModel):
    def __init__(self, modelled_gene_labels: list[str], phi=torch.tanh):
        """Create a cellbox model; a linear ODE with a nonlinear activation function.

        Args:
            modelled_gene_labels (list[str]): list of all modelled genes
            phi: Activation function. Defaults to torch.tanh.
        """
        super().__init__(modelled_gene_labels)
        self.W = nn.Parameter(
            torch.zeros((self.n_genes, self.n_genes), dtype=torch.float32)
        )
        self.decay_params = nn.Parameter(torch.zeros(self.n_genes))
        self.epsilon = nn.Parameter(torch.ones(self.n_genes, dtype=torch.float32))
        self.phi = phi
        self.reset_conditions()

    @property
    def w_masked(self) -> torch.Tensor:
        """Return the weight matrix with the mask applied."""
        return self.W * self.w_mask  # type: ignore

    def reset_conditions(self):
        """Reset the mask to the default values, i.e. no knockouts, only no self-interactions."""
        mask_self_interaction = ~torch.eye(self.n_genes, dtype=torch.bool)
        self.register_buffer("w_mask", mask_self_interaction)

    def set_conditions(self, environment_data: torch.Tensor):
        """Set the conditions for the model. This is used to set the knockout masks for the ODE.

        Currently the environment flags are not used.

        Args:
            environment_data (torch.Tensor): Environment data tensor.
        """
        self.reset_conditions()
        knockout_masks, _ = self.parse_env_data(environment_data)
        new_w_masked = self.w_masked.unsqueeze(0) * knockout_masks  # [B, G, G]
        assert not torch.isnan(new_w_masked).any(), "NaN in w_masked"
        self.register_buffer("w_mask", new_w_masked)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Batch matrix multiplication: w_masked [B, G, G] @ state [B, G, 1] -> change_in_state [B, G]
        input_term = torch.bmm(
            self.w_masked,
            state.unsqueeze(2),  # type: ignore
        ).squeeze(2)
        # Decay always has to be negative, hence the softplus
        decay = nn.functional.softplus(self.decay_params, beta=10)
        gene_expression_changes = self.epsilon * self.phi(input_term) - decay * state
        return gene_expression_changes

    def _set_weight_matrix_to_random(self):
        """Initialise the weight matrix to be sampled from N(0,1)"""
        self.W = nn.Parameter(torch.randn_like(self.W))

    def _set_weight_matrix_to_ones(self):
        """For unit testing mainly"""
        self.W = nn.Parameter(torch.ones_like(self.W))

    def _set_decay_matrix_to_ones(self):
        """For unit testing mainly"""
        self.decay_params = nn.Parameter(torch.ones_like(self.decay_params))

    def _set_epsilon_to_zero(self):
        """For unit testing mainly. Do not fully initialise at 0 because then the gradient can disappear"""
        self.epsilon = nn.Parameter(
            torch.full_like(self.epsilon, 1e-6, dtype=torch.float32)
        )

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
