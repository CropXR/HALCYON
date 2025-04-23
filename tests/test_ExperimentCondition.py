import torch
from mlmmint.Datasets.ExperimentCondition import ExperimentCondition


def test_create_ExperimentCondition():
    experiment_condition = ExperimentCondition(
        all_genes=("G1", "G2", "G3"),
        ko_gene="G2",
        all_environments=("test", "test2"),
        environment="test2",
    )
    assert torch.allclose(
        experiment_condition.as_tensor(), torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
    )
    experiment_condition = ExperimentCondition(
        all_genes=("G1", "G2", "G3"),
        ko_gene=None,
        all_environments=("test", "test2"),
        environment=None,
    )
    assert torch.allclose(
        experiment_condition.as_tensor(), torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    )
