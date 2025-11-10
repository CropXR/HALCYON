from halcyon.Models.MachineLearningModel import MachineLearningModel
from halcyon.RegulatoryNetwork import RegulatoryNetwork


import networkx as nx
import torch
from torch import nn


class CellBox(MachineLearningModel):
    def __init__(
        self,
        modelled_gene_labels: list[str],
        condition_names: tuple[str, ...],
        phi=torch.tanh,
        self_activations: bool = True,
        **kwargs,
    ):
        """Create a cellbox model; a linear ODE with a nonlinear activation function.

        Args:
            modelled_gene_labels (list[str]): list of all modelled genes
            condition_names (tuple[str, ...]): names of the conditions
            phi: Activation function. Defaults to torch.tanh.
        """
        self.phi = phi
        self.self_activations = self_activations
        super().__init__(modelled_gene_labels, condition_names, **kwargs)
        self.W = nn.Parameter(
            torch.randn(
                (self.n_genes, self.n_genes), dtype=torch.float32, device=self.device
            )
        )
        # Init as 1 as decribed in paper
        self.decay_params = nn.Parameter(
            torch.ones(self.n_genes, dtype=torch.float32, device=self.device)
        )
        self.epsilon = nn.Parameter(
            torch.ones(self.n_genes, dtype=torch.float32, device=self.device)
        )

    def get_hyperparams(self):
        out_list = []
        for name, value in zip(
            ["device", "phi", "self_activations"],
            [self.device, self.phi, self.self_activations],
        ):
            out_list.append((name, value))
        return out_list

    @property
    def w_masked(self) -> torch.Tensor:
        """Return the weight matrix with the mask applied."""
        return self.W * self.w_mask  # type: ignore

    def reset_conditions(self):
        """Reset the mask to the default values. In case of no-self interactions, keep the main diagonal zero."""
        if self.self_activations:
            self.w_mask = torch.ones(self.n_genes, device=self.device)
        else:
            self.w_mask = ~torch.eye(self.n_genes, dtype=torch.bool, device=self.device)

    def set_conditions(self, environment_data: torch.Tensor):
        """Set the conditions for the model. This is used to set the knockout masks for the ODE.

        Currently the environment flags are not used.

        Args:
            environment_data (torch.Tensor): Environment data tensor.
        """
        self.reset_conditions()
        knockout_masks, _ = self.parse_env_data(environment_data)
        # knockout_masks is [B, G], need to expand to [B, G, 1] for broadcasting
        knockout_masks_expanded = knockout_masks.unsqueeze(2)  # [B, G, 1]
        new_w_masked = self.w_masked.unsqueeze(0) * knockout_masks_expanded  # [B, G, G]
        assert not torch.isnan(new_w_masked).any(), "NaN in w_masked"
        self.w_mask = new_w_masked

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
        self.W = nn.Parameter(torch.randn_like(self.W, device=self.device))

    def _set_weight_matrix_to_ones(self):
        """For unit testing mainly"""
        self.W = nn.Parameter(torch.ones_like(self.W, device=self.device))

    def _set_decay_matrix_to_ones(self):
        """For unit testing mainly"""
        self.decay_params = nn.Parameter(
            torch.ones_like(self.decay_params, device=self.device)
        )

    def _set_decay_matrix_to_zeros(self):
        """For unit testing mainly"""
        self.decay_params = nn.Parameter(
            torch.zeros_like(self.decay_params, device=self.device)
        )

    def _set_epsilon_to_zero(self):
        """For unit testing mainly. Do not fully initialise at 0 because then the gradient can disappear"""
        self.epsilon = nn.Parameter(
            torch.full_like(self.epsilon, 1e-6, dtype=torch.float32, device=self.device)
        )

    def get_inferred_network(self, threshold: float | None = None) -> RegulatoryNetwork:
        """Return the inferred network

        Args:
            threshold (float, optional): Cutoff to determine from what interaction strength we want
            to predict an edge. Defaults to None. If set to a float, return only edges with weights that have an absolute value larger than this cutoff.

        Returns:
            RegulatoryNetwork
        """
        w_masked = self.w_masked.detach().cpu().numpy()
        G = nx.DiGraph()
        for i, gene_to in enumerate(self.modelled_gene_labels):
            for j, gene_from in enumerate(self.modelled_gene_labels):
                weight = w_masked[i, j]
                if threshold is None or abs(weight) > threshold:
                    G.add_edge(gene_from, gene_to, influence=weight)
        return RegulatoryNetwork(G)
