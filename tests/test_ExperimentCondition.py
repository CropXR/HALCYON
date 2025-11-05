from torch.testing import assert_close
from torch import tensor
from halcyon.Datasets.ExperimentCondition import ExperimentCondition


def test_create_ExperimentCondition():
    experiment_condition = ExperimentCondition(
        all_genes=("G1", "G2", "G3"),
        ko_gene=("G2",),
        all_environments=("test", "test2"),
        environment="test2",
    )
    assert_close(experiment_condition.as_tensor(), tensor([0.0, 1.0, 0.0, 0.0, 1.0]))

    experiment_condition = ExperimentCondition(
        all_genes=("G1", "G2", "G3"),
        ko_gene=None,
        all_environments=("test", "test2"),
        environment=None,
    )
    assert_close(experiment_condition.as_tensor(), tensor([0.0, 0.0, 0.0, 0.0, 0.0]))
