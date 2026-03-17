import logging

import torch

from halcyon.Models.PhoenixModels import NeuralODEWithDecay, PhoenixABC

from torch import nn


class MLP(NeuralODEWithDecay):
    def __init__(
        self,
        modelled_gene_labels: list[str],
        condition_names: tuple[str, ...],
        hidden_size: int,
        nr_layers: int,
        **kwargs,
    ):
        """Create multilayer perceptron

        Args:
            modelled_gene_labels (list[str]): List of genes that the model models.
            condition_names (tuple[str]): names of the conditions that the model models. The length of this is used to create extra input nodes.
            hidden_size (int): Nr of neurons per layer
            nr_layers (int): Number of layers in neural network
        """
        super().__init__(modelled_gene_labels, condition_names, hidden_size, **kwargs)

        layers = [
            nn.Linear(self.n_genes + self.n_conditions, hidden_size),
            nn.Softplus(),
        ]
        for _ in range(nr_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Softplus()])
        layers.append(nn.Linear(hidden_size, self.n_genes))

        self.neural_net = nn.Sequential(*layers)
        self.init_sparse_weights(self.neural_net)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        logging.debug(f"State\t {state}")
        if state.isinf().any() or state.isnan().any():
            raise ValueError("NaN or inf in state of neuralODE")
        nn_input = torch.cat((state, self.env_flags), dim=1)  # type: ignore

        raw_synthesis = self.neural_net(nn_input)
        if self.synthesis_only_positive:
            synthesis = torch.nn.functional.softplus(raw_synthesis)
        else:
            synthesis = raw_synthesis
        synthesis_after_ko = synthesis * self.ko_masks

        decay = self._get_decay(state)

        output = synthesis_after_ko - decay
        logging.debug(f"Output\t {output}")
        return output
