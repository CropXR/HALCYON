from copy import deepcopy
from unittest.mock import MagicMock

import pytest
import torch
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.Losses import MSELossPlusMachineLearningL2
from halcyon.Models.HybridModel import HybridModel
from halcyon.Models.NeuralODE import PhoenixAddAndProdLayers, PhoenixOnlyAddLayers
from halcyon.Models.MechanisticModel import PyTorchODE
from halcyon.Trainer import Trainer
from torch.testing import assert_close


def test_merge_modelled_genes():
    mm_model = MagicMock()
    mm_model.modelled_gene_labels = ["a", "b", "c"]
    mm_model.condition_names = ["cond_1", "cond_2"]
    ml_model = MagicMock()
    ml_model.modelled_gene_labels = ["c", "d"]
    ml_model.condition_names = ["cond_1", "cond_2"]
    hybrid_model = HybridModel(mm_model, ml_model)
    assert set(hybrid_model.modelled_gene_labels) == {"a", "b", "c", "d"}


@pytest.fixture
def expr_data_from_sim(sim_result):
    expr_data = ExpressionDataWithConditions.from_genenetweaver_ko_timeseries_output(
        sim_result.data_path
        / f"{sim_result.network_name}_nonoise_knockout_timeseries.tsv"
    )
    return expr_data


@pytest.fixture
def mech_model_from_sim_result(sim_result, expr_data_from_sim):
    model = PyTorchODE.from_sbml(
        sim_result.data_path / "integration_test_network.xml",
        condition_names=expr_data_from_sim.all_environments,
        from_gnw=True,
    )
    return model


@pytest.fixture
def ml_model_from_sim_result(expr_data_from_sim):
    model = PhoenixAddAndProdLayers(
        expr_data_from_sim.genes, expr_data_from_sim.all_environments, hidden_size=2
    )
    return deepcopy(model)


@pytest.fixture
def hybrid_model(mech_model_from_sim_result, ml_model_from_sim_result):
    return HybridModel(mech_model_from_sim_result, ml_model_from_sim_result)


@pytest.fixture
def hybrid_model_with_zero_ml_influence(
    mech_model_from_sim_result, ml_model_from_sim_result
):
    ml_model_from_sim_result = deepcopy(ml_model_from_sim_result)
    ml_model_from_sim_result._set_per_gene_multipliers_to_zero()
    return HybridModel(mech_model_from_sim_result, ml_model_from_sim_result)


@pytest.fixture
def hybrid_model_mm_no_node_9(mech_model_from_sim_result, ml_model_from_sim_result):
    mech_model_from_sim_result = deepcopy(mech_model_from_sim_result)
    mech_model_from_sim_result.select_genes([f"Node_{i}" for i in range(9)])
    return HybridModel(mech_model_from_sim_result, ml_model_from_sim_result)


@pytest.fixture
def test_batch():
    experiment_data = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    expression_data = torch.tensor(
        [
            [
                [
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                ],
                [
                    6.3392e-01,
                    2.3183e-01,
                    8.4785e-02,
                    3.1007e-02,
                    1.1340e-02,
                    4.1471e-03,
                    1.5166e-03,
                    5.5470e-04,
                    2.0280e-04,
                    7.4200e-05,
                    2.7100e-05,
                    9.9000e-06,
                    3.6000e-06,
                    1.3000e-06,
                    5.0000e-07,
                    2.0000e-07,
                    1.0000e-07,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                ],
                [
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                ],
                [
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                ],
                [
                    9.9317e-01,
                    9.9590e-01,
                    9.9843e-01,
                    9.9947e-01,
                    9.9983e-01,
                    9.9995e-01,
                    9.9998e-01,
                    9.9999e-01,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                    1.0000e00,
                ],
                [
                    1.1174e-02,
                    1.1156e-02,
                    1.1142e-02,
                    1.1135e-02,
                    1.1132e-02,
                    1.1131e-02,
                    1.1131e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                    1.1130e-02,
                ],
                [
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                    2.3878e-01,
                ],
                [
                    2.9734e-01,
                    2.9733e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                    2.9732e-01,
                ],
                [
                    9.9509e-01,
                    9.8513e-01,
                    9.1146e-01,
                    6.3979e-01,
                    2.8439e-01,
                    9.6031e-02,
                    3.1394e-02,
                    1.2691e-02,
                    7.6636e-03,
                    6.3596e-03,
                    6.0277e-03,
                    5.9441e-03,
                    5.9231e-03,
                    5.9179e-03,
                    5.9166e-03,
                    5.9163e-03,
                    5.9162e-03,
                    5.9162e-03,
                    5.9161e-03,
                    5.9161e-03,
                    5.9161e-03,
                ],
            ],
            [
                [
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                    7.2333e-01,
                ],
                [
                    6.3392e-01,
                    6.3358e-01,
                    6.3338e-01,
                    6.3325e-01,
                    6.3317e-01,
                    6.3313e-01,
                    6.3310e-01,
                    6.3308e-01,
                    6.3307e-01,
                    6.3306e-01,
                    6.3306e-01,
                    6.3306e-01,
                    6.3306e-01,
                    6.3306e-01,
                    6.3306e-01,
                    6.3306e-01,
                    6.3306e-01,
                    6.3306e-01,
                    6.3305e-01,
                    6.3305e-01,
                    6.3305e-01,
                ],
                [
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                    9.4803e-01,
                ],
                [
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                    7.4355e-01,
                ],
                [
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                    1.3590e-01,
                ],
                [
                    9.9317e-01,
                    9.9318e-01,
                    9.9319e-01,
                    9.9319e-01,
                    9.9319e-01,
                    9.9319e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                    9.9320e-01,
                ],
                [
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                    1.1174e-02,
                ],
                [
                    2.3878e-01,
                    7.4420e-02,
                    2.3194e-02,
                    7.2289e-03,
                    2.2530e-03,
                    7.0220e-04,
                    2.1890e-04,
                    6.8200e-05,
                    2.1300e-05,
                    6.6000e-06,
                    2.1000e-06,
                    6.0000e-07,
                    2.0000e-07,
                    1.0000e-07,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    2.9734e-01,
                    1.9087e-01,
                    9.0266e-02,
                    4.8217e-02,
                    3.3836e-02,
                    2.9262e-02,
                    2.7851e-02,
                    2.7421e-02,
                    2.7292e-02,
                    2.7252e-02,
                    2.7241e-02,
                    2.7237e-02,
                    2.7236e-02,
                    2.7236e-02,
                    2.7236e-02,
                    2.7236e-02,
                    2.7236e-02,
                    2.7236e-02,
                    2.7236e-02,
                    2.7236e-02,
                    2.7236e-02,
                ],
                [
                    9.9509e-01,
                    9.9508e-01,
                    9.9508e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                    9.9507e-01,
                ],
            ],
        ]
    )
    return experiment_data, expression_data


def test_hybrid_does_perfect_fit_if_ml_weights_zero(
    expr_data_from_sim, test_batch, hybrid_model_with_zero_ml_influence
):
    # Set all per_gene_multipliers to 0 and check if MM model matches data perfectly
    my_trainer = Trainer(
        hybrid_model_with_zero_ml_influence, expr_data_from_sim, batch_size=2
    )
    experiment_data, expression_data = my_trainer.preprocess_batch(test_batch)
    output = my_trainer._calculate_model_prediction_at_conditions(
        experiment_data, expression_data
    )
    assert_close(expression_data, output, atol=1e-3, rtol=1e-3)
    loss = my_trainer.calculate_loss_expression_vs_pred(expression_data, output)
    assert_close(loss, torch.tensor(0.0, device=my_trainer.model.device))


def test_most_genes_perfect_fit_if_only_remove_gene_9(
    expr_data_from_sim,
    hybrid_model_mm_no_node_9,
    test_batch,
):
    hybrid_model_mm_no_node_9 = deepcopy(hybrid_model_mm_no_node_9)
    hybrid_model_mm_no_node_9._ml._set_per_gene_multipliers_to_zero()
    my_trainer = Trainer(
        hybrid_model_mm_no_node_9,
        expr_data_from_sim,
        batch_size=2,
    )
    experiment_data, expression_data = my_trainer.preprocess_batch(test_batch)
    output = my_trainer._calculate_model_prediction_at_conditions(
        experiment_data, expression_data
    )
    # First 9 genes will match perfectly
    assert_close(expression_data[:, :9, :], output[:, :9, :], atol=1e-3, rtol=1e-3)
    loss = my_trainer.calculate_loss_expression_vs_pred(expression_data, output)
    assert_close(
        loss, torch.tensor(0.0382, device=my_trainer.model.device), atol=1e-3, rtol=1e-3
    )


def test_l2_loss(expr_data_from_sim, hybrid_model_mm_no_node_9, test_batch):
    """Comment: this might fail if ran alone, but passes if run together with the other tests. Have looked into it but gave up after a while. It's due to different bias (weights) in the ML model though"""
    my_l2_loss = MSELossPlusMachineLearningL2(lambda_param=1)

    my_trainer = Trainer(
        hybrid_model_mm_no_node_9,
        expr_data_from_sim,
        batch_size=2,
        loss=my_l2_loss,
    )
    loss = my_trainer._calculate_loss_from_batch(test_batch)
    assert_close(
        loss,
        torch.tensor(13143.2842 + 0.0096, device=my_trainer.model.device),
        atol=1e-4,
        rtol=1e-4,
    )
    # Changing lambda
    my_trainer.loss_fn = MSELossPlusMachineLearningL2(lambda_param=0.5)
    loss = my_trainer._calculate_loss_from_batch(test_batch)
    assert_close(
        loss,
        torch.tensor(13143.2842 + 0.0096 * 0.5, device=my_trainer.model.device),
        atol=1e-4,
        rtol=1e-4,
    )
    # Checking that changing ML weights changes loss, too.
    my_trainer.model._ml._set_per_gene_multipliers_to_zero()
    loss = my_trainer._calculate_loss_from_batch(test_batch)
    assert_close(
        loss,
        torch.tensor(0.0430, device=my_trainer.model.device),
        atol=1e-4,
        rtol=1e-4,
    )


def test_hybrid_model_gene_split(hybrid_model_mm_no_node_9):

    assert_close(
        hybrid_model_mm_no_node_9._mm_indices,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        check_device=False,
    )
    assert_close(
        hybrid_model_mm_no_node_9._ml_indices,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        check_device=False,
    )
    assert [
        hybrid_model_mm_no_node_9.modelled_gene_labels[i]
        for i in hybrid_model_mm_no_node_9._mm_indices.tolist()
    ] == [f"Node_{i}" for i in range(9)]
