from halcyon.Models.GeneExpressionModel import GeneExpressionModel
import torch
from torch import nn


class CellBox(GeneExpressionModel):
    def __init__(
        self,
        modelled_gene_labels: list[str],
        condition_names: tuple[str, ...],
        phi=torch.tanh,
        self_activations=False,
        **kwargs,
    ):
        super().__init__(modelled_gene_labels, condition_names, **kwargs)
        self.phi = phi
        self.self_activations = self_activations
        # Init parameters as decribed in paper
        self.W = nn.Parameter(
            torch.normal(
                mean=0.01,
                std=1,
                size=(self.n_genes, self.n_genes),
                dtype=torch.float32,
            )
        )
        self.decay_params = nn.Parameter(torch.ones(self.n_genes, dtype=torch.float32))
        self.epsilon = nn.Parameter(torch.ones(self.n_genes, dtype=torch.float32))

        # In case of no-self interactions, keep the main diagonal zero.
        if self.self_activations:
            w_mask = torch.ones(self.n_genes)
        else:
            w_mask = ~torch.eye(self.n_genes, dtype=torch.bool)

        self.register_buffer("w_mask", w_mask)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        masked_w = self.W * self.w_mask
        # Batch matrix multiplication: w_masked [B, G, G] @ state [B, G, 1] -> change_in_state [B, G]
        input_term = torch.bmm(
            masked_w.unsqueeze(0).expand(state.shape[0], -1, -1),
            state.unsqueeze(2),  # type: ignore
        ).squeeze(2)
        synthesis = self.ko_masks * (self.epsilon * self.phi(input_term))
        # Decay always has to be negative, hence the softplus
        decay = nn.functional.softplus(self.decay_params * state)
        gene_expression_changes = synthesis - decay
        return gene_expression_changes

    def get_hyperparams(self):
        out_list = []
        for name, value in zip(
            ["device", "phi", "self_activations"],
            [self.device, self.phi.__name__, self.self_activations],
        ):
            out_list.append((name, value))
        return out_list
