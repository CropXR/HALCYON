from abc import ABC, abstractmethod
import logging

import torch
from torch import nn

from halcyon.Models.ActivationFunctions import (
    PhoenixProdActivation,
    PhoenixSumActivation,
)
from halcyon.Models.GeneExpressionModel import GeneExpressionModel
from halcyon.Datasets.RegulatoryNetwork import RegulatoryNetwork


class NeuralODEWithDecay(GeneExpressionModel, ABC):
    def __init__(
        self,
        modelled_gene_labels: list[str],
        condition_names: tuple[str, ...],
        hidden_size: int,
        synthesis_only_positive: bool = False,
        **kwargs,
    ):
        super().__init__(modelled_gene_labels, condition_names, **kwargs)
        self.hidden_size = hidden_size
        self.decay_params = nn.Parameter(torch.rand(self.n_genes))
        self.register_buffer(
            "synthesis_only_positive", torch.tensor(synthesis_only_positive)
        )

    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        raise NotImplementedError

    def _get_decay(self, state):
        return nn.functional.relu(self.decay_params * state)

    def init_sparse_weights(self, model: nn.Sequential, sparsity=0.95):
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.sparse_(layer.weight, sparsity=sparsity, std=0.05)

    def get_hyperparams(self):
        out_list = []
        for name, value in zip(
            ["device", "hidden_size", "synthesis_only_positive"],
            [self.device, self.hidden_size, self.synthesis_only_positive],
        ):
            out_list.append((name, value))

        return out_list


class PhoenixABC(NeuralODEWithDecay, ABC):
    def __init__(
        self,
        modelled_gene_labels: list[str],
        condition_names: tuple[str, ...],
        hidden_size: int,
        synthesis_only_positive: bool = False,
        **kwargs,
    ):
        """Phoenix ABC. Here this is like the NeuralODEWithDecay but also provides a per-gene multiplier (like in the PHOENIX paper)"""
        super().__init__(
            modelled_gene_labels,
            condition_names,
            hidden_size,
            synthesis_only_positive,
            **kwargs,
        )
        self.per_gene_multipliers = nn.Parameter(torch.ones(self.n_genes))

    def _set_raw_per_gene_importance(self, value=-50):
        """Provide per gene importance score

        Args:
            value (float, optional): Raw (i.e. before applying softplus) importance score to use for each gene. Defaults to -50.
        """
        with torch.no_grad():
            self.per_gene_multipliers.fill_(value)


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
        synthesis = self.neural_net(nn_input) * self.ko_masks
        if self.synthesis_only_positive:
            synthesis = torch.nn.functional.softplus(synthesis)
        output = (synthesis - decay) * torch.nn.functional.softplus(self.per_gene_multipliers)  # type: ignore
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

        synthesis = self.combine_layer_nn(combined_nn_input) * self.ko_masks
        decay = self._get_decay(state)
        if self.synthesis_only_positive:
            synthesis = torch.nn.functional.softplus(synthesis)
        output = (synthesis - decay) * torch.nn.functional.softplus(self.per_gene_multipliers)  # type: ignore
        logging.debug(f"Output\t {output}")
        return output
