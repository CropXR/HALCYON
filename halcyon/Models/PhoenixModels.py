from abc import ABC, abstractmethod
from halcyon.Models.ActivationFunctions import (
    PhoenixProdActivation,
    PhoenixSumActivation,
)
from halcyon.Models.MachineLearningModel import MachineLearningModel
from halcyon.RegulatoryNetwork import RegulatoryNetwork


import torch
from torch import nn

import logging


class PhoenixABC(MachineLearningModel, ABC):
    def __init__(
        self,
        modelled_gene_labels: list[str],
        condition_names: tuple[str, ...],
        hidden_size: int,
        **kwargs,
    ):
        """Class that is inherited, so shouldn't be called on its own."""
        super().__init__(modelled_gene_labels, condition_names, **kwargs)
        self.hidden_size = hidden_size
        self.decay_params = nn.Parameter(torch.rand(self.n_genes))
        self.per_gene_multipliers = nn.Parameter(torch.rand(self.n_genes) * 0.1)

    def init_sparse_weights(self, model: nn.Sequential, sparsity=0.95):
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.sparse_(layer.weight, sparsity=sparsity, std=0.05)

    def _get_decay(self, state):
        return nn.functional.relu(self.decay_params * state)

    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        raise NotImplementedError

    def _set_per_gene_multipliers_to_zero(self):
        with torch.no_grad():
            self.per_gene_multipliers.zero_()

    @abstractmethod
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pass

    def get_hyperparams(self):
        out_list = []
        for name, value in zip(
            ["device", "hidden_size"], [self.device, self.hidden_size]
        ):
            out_list.append((name, value))

        return out_list


class PhoenixOnlyAddLayers(PhoenixABC):
    def __init__(
        self,
        modelled_gene_labels: list[str],
        condition_names: tuple[str, ...],
        hidden_size: int,
        **kwargs,
    ):
        """Create Phoenix model; a neural network that takes the gene expression and the environment as input and returns the change in gene expression. This version only has additive layers, so no interactive effects are modelled.

        Args:
            modelled_gene_labels (list[str]): List of genes that the model models.
            condition_names (tuple[str]): names of the conditions that the model models. The length of this is used to create extra input nodes.
        """
        super().__init__(modelled_gene_labels, condition_names, hidden_size, **kwargs)
        self.neural_net = nn.Sequential(
            PhoenixSumActivation(),
            nn.Linear(self.n_genes + self.n_conditions, self.hidden_size),
            nn.Linear(self.hidden_size, self.n_genes),
        )
        self.init_sparse_weights(self.neural_net)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        logging.debug(f"State\t {state}")
        if state.isinf().any() or state.isnan().any():
            raise ValueError("NaN or inf in state of neuralODE")
        nn_input = torch.cat((state, self.env_flags), dim=1)  # type: ignore
        decay = self._get_decay(state)
        output = (self.neural_net(nn_input) * self.ko_masks - decay) * torch.nn.functional.relu(self.per_gene_multipliers)  # type: ignore
        logging.debug(f"Output\t {output}")
        return output


class PhoenixAddAndProdLayers(PhoenixABC):
    def __init__(
        self,
        modelled_gene_labels: list[str],
        condition_names: tuple[str, ...],
        hidden_size: int,
        **kwargs,
    ):
        """Create Phoenix model; a neural network that takes the gene expression and the environment as input and returns the change in gene expression.
        This version has both additive and multiplicative layers.

        Args:
            modelled_gene_labels (list[str]): List of genes that the model models.
            condition_names (tuple[str]): names of the conditions that the model models. The length of this is used to create extra input nodes.
            hidden_size (int): size of the hidden state.
        """
        super().__init__(modelled_gene_labels, condition_names, hidden_size, **kwargs)
        self.sum_nn = nn.Sequential(
            PhoenixSumActivation(),
            nn.Linear(self.n_genes + self.n_conditions, self.hidden_size),
        )
        self.prod_nn = nn.Sequential(
            PhoenixProdActivation(),
            nn.Linear(self.n_genes + self.n_conditions, self.hidden_size),
        )

        self.combine_layer_nn = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.n_genes)
        )
        for model in [self.sum_nn, self.prod_nn, self.combine_layer_nn]:
            self.init_sparse_weights(model)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        logging.debug(f"State\t {state}")
        if state.isinf().any() or state.isnan().any():
            raise ValueError("NaN or inf in state of neuralODE")
        first_layer_input = torch.cat((state, self.env_flags), dim=1)  # type: ignore

        sum_out = self.sum_nn(first_layer_input)
        prod_out = torch.exp(self.prod_nn(first_layer_input))
        combined_nn_input = torch.cat((sum_out, prod_out), dim=1)

        decay = self._get_decay(state)
        output = (self.combine_layer_nn(combined_nn_input) * self.ko_masks - decay) * torch.nn.functional.relu(self.per_gene_multipliers)  # type: ignore
        logging.debug(f"Output\t {output}")
        return output
