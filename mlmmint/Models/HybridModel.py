from mlmmint.Models.MechanisticModel import MechanisticModel
from mlmmint.Models.Model import Model


class HybridModel(Model):
    def __init__(self, mm_model: MechanisticModel, ml_model: Model):
        all_modelled_genes = set(mm_model.modelled_gene_labels) | set(
            ml_model.modelled_gene_labels
        )
        super().__init__(modelled_gene_labels=list(all_modelled_genes))
        self.mechanistic = mm_model
        self.ml = ml_model

    def forward(self):
        raise NotImplementedError

    def integrate(self):
        raise NotImplementedError
