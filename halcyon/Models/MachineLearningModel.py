import torch
from halcyon.Models.Model import Model
from abc import abstractmethod, ABC

from halcyon.RegulatoryNetwork import RegulatoryNetwork


class MachineLearningModel(Model, ABC):
    def __init__(self, modelled_gene_labels, condition_names, **kwargs):
        super().__init__(modelled_gene_labels, **kwargs)
        self.n_conditions = len(condition_names)

    @abstractmethod
    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        """Extract the inferred network"""
        pass

    def parse_env_data(
        self, environment_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given the environment data as a tensor, split that into the knockout masks and the environment flags.

        Can be used by children classes to proceed with these KO and env flags.
        Args:
            environment_data (torch.Tensor): tensor of [n_batch, n_genes + n_conditions].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple of knockout masks and environment flags.
        """
        assert (
            environment_data.shape[1] == self.n_genes + self.n_conditions
        ), "Environment data should be of shape (B, G + C) where B is the batch size, G is the number of genes and C is the number of conditions"
        knockout_flags = environment_data[:, : self.n_genes]
        env_flags = environment_data[:, self.n_genes :]
        knockout_masks = 1.0 - knockout_flags.unsqueeze(2)
        return knockout_masks, env_flags
