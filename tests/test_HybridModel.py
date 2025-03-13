from unittest.mock import MagicMock
from mlmmint.Models.HybridModel import HybridModel


def test_merge_modelled_genes():
    mm_model = MagicMock()
    mm_model.modelled_gene_labels = ["a", "b", "c"]
    ml_model = MagicMock()
    ml_model.modelled_gene_labels = ["c", "d"]
    hybrid_model = HybridModel(mm_model, ml_model)
    assert set(hybrid_model.modelled_gene_labels) == {"a", "b", "c", "d"}
