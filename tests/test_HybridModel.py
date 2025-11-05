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


def test_hybrid_model_gene_split_THISISOLD(get_synthetic_data):
    gt_regulatory_network, interactions, expression_dataset = get_synthetic_data
    subset_regulatory_network = gt_regulatory_network.select_subset(["G1", "G4", "G5"])
    env_conditions, expressions = expression_dataset[0]

    # imitate batch
    env_conditions, expressions = env_conditions.unsqueeze(0), expressions.unsqueeze(0)

    ml_model = PhoenixOnlyAddLayers(
        expression_dataset.genes,
        expression_dataset.all_environments,
        hidden_size=2,
    )
    mm_model = TorchLinearODE(subset_regulatory_network)
    hybrid_model = HybridModel(mm_model, ml_model)
    env_conditions = env_conditions.to(hybrid_model.device)
    expressions = expressions.to(hybrid_model.device)
    # Ensure that they handle the different order of genes well
    assert_close(
        hybrid_model.mm_indices, torch.tensor([0, 3, 4], device=hybrid_model.device)
    )
    assert_close(
        hybrid_model.ml_indices,
        torch.tensor([0, 1, 2, 3, 4], device=hybrid_model.device),
    )
    assert [
        hybrid_model.modelled_gene_labels[i] for i in hybrid_model.mm_indices.tolist()
    ] == ["G1", "G4", "G5"]
    hybrid_model.set_conditions(env_conditions)
    assert_close(
        hybrid_model._get_mm_forward(0, expressions[:, :, 0]),
        torch.tensor([[0.1000, 0.0000, -0.0800]], device=hybrid_model.device),
    )

    one_forward = hybrid_model.forward(
        t=torch.tensor([0], device=hybrid_model.device), state=expressions[:, :, 0]
    )
    assert one_forward.shape == (1, 5)

    trainer = Trainer(hybrid_model, expression_dataset)

    integration_output = trainer.integrate_model(expressions[:, :, 0])
    # Check if at least shape and initial gene expressions match
    assert tuple(integration_output.shape) == (1, 5, 4)
    assert_close(
        integration_output[0, :, 0],
        torch.tensor(
            [0.1000, 0.2000, 0.3000, 0.4000, 0.5000], device=hybrid_model.device
        ),
    )


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
    return model


@pytest.fixture
def hybrid_model_with_ml_zero_weights(
    mech_model_from_sim_result, ml_model_from_sim_result
):
    ml_model_from_sim_result._set_per_gene_multipliers_to_zero()
    return HybridModel(mech_model_from_sim_result, ml_model_from_sim_result)


@pytest.fixture
def hybrid_model_with_ml_zero_weights_and_mm_missing_node_9(
    mech_model_from_sim_result, ml_model_from_sim_result
):
    ml_model_from_sim_result._set_per_gene_multipliers_to_zero()
    mech_model_from_sim_result.select_genes([f"Node_{i}" for i in range(9)])
    return HybridModel(mech_model_from_sim_result, ml_model_from_sim_result)


def test_hybrid_does_perfect_fit_if_ml_weights_zero(
    expr_data_from_sim, hybrid_model_with_ml_zero_weights
):
    # Set all per_gene_multipliers to 0 and check if MM model matches data perfectly
    my_trainer = Trainer(
        hybrid_model_with_ml_zero_weights, expr_data_from_sim, batch_size=2
    )
    test_batch = next(iter(my_trainer.dataloader))
    experiment_data, expression_data = my_trainer.preprocess_batch(test_batch)
    output = my_trainer._calculate_model_prediction_at_conditions(
        experiment_data, expression_data
    )
    assert torch.allclose(expression_data, output, atol=1e-3, rtol=1e-3)
    loss = my_trainer.calculate_loss_expression_vs_pred(expression_data, output)
    assert torch.allclose(loss, torch.tensor([0.0], device=my_trainer.model.device))


def test_most_genes_perfect_fit_if_only_remove_gene_9(
    expr_data_from_sim, hybrid_model_with_ml_zero_weights_and_mm_missing_node_9
):
    my_trainer = Trainer(
        hybrid_model_with_ml_zero_weights_and_mm_missing_node_9,
        expr_data_from_sim,
        batch_size=2,
    )
    test_batch = next(iter(my_trainer.dataloader))
    experiment_data, expression_data = my_trainer.preprocess_batch(test_batch)
    output = my_trainer._calculate_model_prediction_at_conditions(
        experiment_data, expression_data
    )
    # First 9 genes will match perfectly
    assert torch.allclose(
        expression_data[:, :9, :], output[:, :9, :], atol=1e-3, rtol=1e-3
    )
    loss = my_trainer.calculate_loss_expression_vs_pred(expression_data, output)
    # Loss changes massively for some reasons -> Not replicable?! TODO
    assert torch.allclose(loss, torch.tensor([0.0382], device=my_trainer.model.device))


def test_l2_loss(expr_data_from_sim, hybrid_model_with_ml_zero_weights):
    my_l2_loss = MSELossPlusMachineLearningL2(lambda_param=0.5)
    my_trainer = Trainer(
        hybrid_model_with_ml_zero_weights,
        expr_data_from_sim,
        batch_size=2,
        loss=my_l2_loss,
    )
    test_batch = next(iter(my_trainer.dataloader))
    loss = my_trainer._calculate_loss_from_batch(test_batch)
    assert torch.allclose(
        torch.tensor(0.0043, device=my_trainer.model.device), loss, atol=1e-3, rtol=1e-3
    )


def test_hybrid_model_gene_split(
    expr_data_from_sim, hybrid_model_with_ml_zero_weights_and_mm_missing_node_9
):
    hybrid_model = hybrid_model_with_ml_zero_weights_and_mm_missing_node_9

    torch.testing.assert_close(
        hybrid_model.mm_indices,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        check_device=False,
    )
    torch.testing.assert_close(
        hybrid_model.ml_indices,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        check_device=False,
    )
    assert [
        hybrid_model.modelled_gene_labels[i] for i in hybrid_model.mm_indices.tolist()
    ] == [f"Node_{i}" for i in range(9)]

    # my_l2_loss = MSELossPlusMachineLearningL2(lambda_param=0.5)
    # my_trainer = Trainer(
    #     hybrid_model, expr_data_from_sim, batch_size=2, loss=my_l2_loss
    # )
    # test_batch = next(iter(my_trainer.dataloader))
    # loss = my_trainer._calculate_loss_from_batch(test_batch)
    # assert torch.allclose(
    #     torch.tensor(0.0043, device=my_trainer.model.device), loss, atol=1e-3, rtol=1e-3
    # )
