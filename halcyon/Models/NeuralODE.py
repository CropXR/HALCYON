from halcyon.Models.MachineLearningModel import MachineLearningModel
from halcyon.RegulatoryNetwork import RegulatoryNetwork


import torch
from torch import nn


import logging


class PhoenixSumActivation(nn.Module):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return (y - 0.5) / (1 + torch.abs(y - 0.5))


class NeuralODE(MachineLearningModel):
    def __init__(
        self,
        modelled_gene_labels: list[str],
        condition_names: tuple[str, ...],
        activation_function=PhoenixSumActivation,
    ):
        """Create neuralODE, i.e. a neural network that takes the gene expression and the environment as input and returns the change in gene expression.

        Args:
            modelled_gene_labels (list[str]): List of genes that the model models.
            condition_names (tuple[str]): names of the conditions that the model models. The length of this is used to create extra input nodes for the neural network.
        """
        super().__init__(modelled_gene_labels)
        self.n_conditions = len(condition_names)
        # Architecture inspired by PHOENIX
        self.neural_net = nn.Sequential(
            activation_function(),
            nn.Linear(self.n_genes + self.n_conditions, self.n_genes),
        )
        self.decay_params = nn.Parameter(
            torch.full(size=(self.n_genes,), fill_value=0.1)
        )
        self.per_gene_multipliers = nn.Parameter(
            torch.full(size=(self.n_genes,), fill_value=1.0)
        )
        self.reset_conditions()

    def set_conditions(self, env_data: torch.Tensor):
        """Use the env_data tensor to set the condition (i.e. knockouts and environment flags) for the model.

        On each forward pass the environment flags are used as input to the neural network, and the knockout masks are used to mask the output of the neural network.
        Args:
            env_data (torch.Tensor): Tensor of shape (B, G + C) where B is the batch size, G is the number of genes and C is the number of conditions.
        """
        self.reset_conditions()
        assert (
            env_data.shape[1] == self.n_genes + self.n_conditions
        ), "Environment data should be of shape (B, G + C) where B is the batch size, G is the number of genes and C is the number of conditions"
        # Knockouts should be used as mask after the forwards pass
        ko_masks, env_flags = self.parse_env_data(env_data)
        self.register_buffer("env_flags", env_flags)
        # Environments should be appended to the forward pass but not returned (Because they do not change over time)
        self.register_buffer("ko_masks", ko_masks)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        logging.debug(f"State\t {state}")
        if state.isinf().any() or state.isnan().any():
            raise ValueError("NaN or inf in state of neuralODE")
        nn_input = torch.cat((state, self.env_flags), dim=1)  # type: ignore
        decay = self._get_decay(state)
        output = (self.neural_net(nn_input) * self.ko_masks.squeeze(2) - decay) * torch.nn.functional.relu(self.per_gene_multipliers)  # type: ignore
        logging.debug(f"Output\t {output}")
        return output

    def _get_decay(self, state):
        return nn.functional.relu(self.decay_params * state)

    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        raise NotImplementedError

    def reset_conditions(self):
        self.register_buffer("ko_masks", torch.ones(self.n_genes))  # type: ignore
