from mlmmint.Models.MechanisticModel import MechanisticModel
from mlmmint.Models.Model import Model
import torch


class HybridModel(Model):
    def __init__(self, mm_model: MechanisticModel, ml_model: Model):
        all_modelled_genes = list(
            set(mm_model.modelled_gene_labels) | set(ml_model.modelled_gene_labels)
        )
        super().__init__(modelled_gene_labels=list(all_modelled_genes))
        self.mechanistic = mm_model
        self.ml = ml_model

        # Precompute indices
        self.mm_indices = torch.tensor(
            [self.modelled_gene_labels.index(g) for g in mm_model.modelled_gene_labels]
        )
        self.ml_indices = torch.tensor(
            [self.modelled_gene_labels.index(g) for g in ml_model.modelled_gene_labels]
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        mm_output = self._get_mm_forward(t, state)
        ml_output = self._get_ml_forward(t, state)

        output = torch.zeros_like(state)
        output.index_add_(dim=0, index=self.mm_indices, source=mm_output)
        output.index_add_(dim=0, index=self.ml_indices, source=ml_output)
        return output

    def _get_ml_forward(self, t, state):
        ml_input = torch.index_select(state, dim=0, index=self.ml_indices)
        ml_output = self.ml.forward(t, ml_input)
        return ml_output

    def _get_mm_forward(self, t, state):
        mm_input = torch.index_select(state, dim=0, index=self.mm_indices)
        mm_output = self.mechanistic.forward(t, mm_input)
        return mm_output
