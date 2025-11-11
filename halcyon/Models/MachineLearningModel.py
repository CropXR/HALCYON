from typing import Generator
import torch
from halcyon.Models.Model import Model
from abc import abstractmethod, ABC

from halcyon.RegulatoryNetwork import RegulatoryNetwork


class MachineLearningModel(Model, ABC):
    def __init__(
        self, modelled_gene_labels: list[str], condition_names: list[str], **kwargs
    ):
        super().__init__(modelled_gene_labels, condition_names, **kwargs)

    @abstractmethod
    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        """Extract the inferred network"""
        pass
