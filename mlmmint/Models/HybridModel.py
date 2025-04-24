import logging
import torch

from mlmmint.Models.MachineLearningModel import MachineLearningModel
from mlmmint.Models.MechanisticModel import MechanisticModel
from mlmmint.Models.Model import Model
from mlmmint.RegulatoryNetwork import RegulatoryNetwork


class HybridModel(Model):
    def __init__(self, mm: MechanisticModel, ml: MachineLearningModel):
        """
        Initialize the HybridModel with a mechanistic model and a machine learning model.
        Args:
            mm (MechanisticModel): The mechanistic model instance.
            ml (Model): The machine learning model instance.
        Attributes:
            mm (MechanisticModel): The mechanistic model instance.
            ml (Model): The machine learning model instance.
            mm_indices (torch.Tensor): Precomputed indices for the mechanistic model genes.
            ml_indices (torch.Tensor): Precomputed indices for the machine learning model genes.
        """
        all_modelled_genes = list(
            set(mm.modelled_gene_labels) | set(ml.modelled_gene_labels)
        )
        super().__init__(modelled_gene_labels=list(all_modelled_genes))
        self.mm = mm
        self.ml = ml

        # Precompute indices
        self.mm_indices = torch.tensor(
            [self.modelled_gene_labels.index(g) for g in mm.modelled_gene_labels]
        )
        self.ml_indices = torch.tensor(
            [self.modelled_gene_labels.index(g) for g in ml.modelled_gene_labels]
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Perform forward pass

        Args:
            t (torch.Tensor): Time point to pass to forward pass
            state (torch.Tensor): Gene expressions at that time point. Should be in shape [batch_size, n_genes].

        Returns:
            torch.Tensor: Change in expressions of genes. Again in shape [batch_size, n_genes].
        """
        mm_output = self._get_mm_forward(t, state)
        ml_output = self._get_ml_forward(t, state)

        output = torch.zeros_like(state)
        output.index_add_(dim=1, index=self.mm_indices, source=mm_output)
        output.index_add_(dim=1, index=self.ml_indices, source=ml_output)
        return output

    def _get_ml_forward(self, t, state) -> torch.Tensor:
        ml_input = state[:, self.ml_indices]
        ml_output = self.ml.forward(t, ml_input)
        return ml_output

    def _get_mm_forward(self, t, state) -> torch.Tensor:
        mm_input = state[:, self.mm_indices]
        mm_output = self.mm.forward(t, mm_input)
        return mm_output

    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        """Get the inferred network

        Returns:
            RegulatoryNetwork: Inferred network from data
        """
        logging.warning(
            "Now just use the ML-model network inference, but later on combine this with MM?"
        )
        return self.ml.get_inferred_network(threshold=threshold)

    def set_conditions(self, env_conditions: torch.Tensor):
        self.mm.set_conditions(env_conditions)
        self.ml.set_conditions(env_conditions)
