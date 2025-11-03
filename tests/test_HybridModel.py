from unittest.mock import MagicMock

import pytest
import torch
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.Models.HybridModel import HybridModel
from halcyon.Models.NeuralODE import PhoenixAddAndProdLayers, PhoenixOnlyAddLayers
from halcyon.Models.MechanisticModel import PyTorchODE
from halcyon.Trainer import Trainer
from tests.create_dummy_data import get_synthetic_data

from tests.test_MechanisticModel import test_mechanistic_model
from tests.test_GNWSimRunner import sim_result


def test_merge_modelled_genes():
    mm_model = MagicMock()
    mm_model.modelled_gene_labels = ["a", "b", "c"]
    mm_model.condition_names = ["cond_1", "cond_2"]
    ml_model = MagicMock()
    ml_model.modelled_gene_labels = ["c", "d"]
    ml_model.condition_names = ["cond_1", "cond_2"]
    hybrid_model = HybridModel(mm_model, ml_model)
    assert set(hybrid_model.modelled_gene_labels) == {"a", "b", "c", "d"}


def test_hybrid_model_gene_split():
    gt_regulatory_network, interactions, expression_dataset = get_synthetic_data()
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
    assert torch.allclose(
        hybrid_model.mm_indices, torch.tensor([0, 3, 4], device=hybrid_model.device)
    )
    assert torch.allclose(
        hybrid_model.ml_indices,
        torch.tensor([0, 1, 2, 3, 4], device=hybrid_model.device),
    )
    assert [
        hybrid_model.modelled_gene_labels[i] for i in hybrid_model.mm_indices.tolist()
    ] == ["G1", "G4", "G5"]
    hybrid_model.set_conditions(env_conditions)
    assert torch.allclose(
        hybrid_model._get_mm_forward(0, expressions[:, :, 0]),
        torch.tensor([0.1000, 0.0000, -0.0800], device=hybrid_model.device),
    )

    one_forward = hybrid_model.forward(
        t=torch.tensor([0], device=hybrid_model.device), state=expressions[:, :, 0]
    )
    assert one_forward.shape == (1, 5)

    trainer = Trainer(hybrid_model, expression_dataset)

    integration_output = trainer.integrate_model(expressions[:, :, 0])
    # Check if at least shape and initial gene expressions match
    assert tuple(integration_output.shape) == (1, 5, 4)
    assert torch.allclose(
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
        from_GNW=True,
    )
    return model


@pytest.fixture
def ml_model_from_sim_result(expr_data_from_sim):
    model = PhoenixAddAndProdLayers(
        expr_data_from_sim.genes, expr_data_from_sim.all_environments, hidden_size=2
    )
    return model


def test_mm_and_ml_merging(
    expr_data_from_sim, mech_model_from_sim_result, ml_model_from_sim_result
):
    # Set all per_gene_multipliers to 0 and check if MM model matches data perfectly
    ml_model_from_sim_result._set_per_gene_multipliers_to_zero()
    hybrid_model = HybridModel(mech_model_from_sim_result, ml_model_from_sim_result)

    my_trainer = Trainer(hybrid_model, expr_data_from_sim, batch_size=2)
    test_batch = next(iter(my_trainer.dataloader))
    experiment_data, expression_data = my_trainer.preprocess_batch(test_batch)
    output = my_trainer._calculate_model_prediction_at_conditions(
        experiment_data, expression_data
    )

    assert torch.allclose(expression_data, output, atol=1e-3, rtol=1e-3)
