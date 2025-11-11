from copy import deepcopy
from pathlib import Path
import random
from typing import Tuple
import networkx as nx
from halcyon.Models.HybridModel import HybridModel
from halcyon.Models.MechanisticModel import PyTorchODE
import numpy as np

import pytest
import torch
from halcyon.Datasets.ExperimentCondition import ExperimentCondition
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions

from halcyon.FittedTrainerList import FittedTrainerList
from halcyon.Models.CellBox import CellBox
from halcyon.Models.PhoenixModels import PhoenixAddAndProdLayers, PhoenixOnlyAddLayers
from halcyon.RegulatoryNetwork import RegulatoryNetwork

from halcyon.Simulators.GNWSimRunner import GNWSimRunner, SimulationResult
from halcyon.Trainer import Trainer


@pytest.fixture(scope="session", autouse=True)
def set_random_seed():
    seed = 174
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture
def expression_data_from_gnw():
    expressions = ExpressionDataWithConditions.from_genenetweaver_ko_timeseries_output(
        Path(
            "tests/expected_output/integration_test_network_nonoise_knockout_timeseries.tsv"
        )
    )
    return expressions


@pytest.fixture
def sim_result() -> SimulationResult:
    gnw_sim = GNWSimRunner(
        settings_path=Path("setting_files/gnw_testing_runs_settings.txt")
    )
    generated_data_path = Path("tests/generated_test_data")

    sim_result = gnw_sim.run_simulation(
        n_genes=10,
        sparsity=0.1,
        num_modules=2,
        network_name="integration_test_network",
        output_path=generated_data_path,
    )
    return sim_result


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
    return HybridModel(
        mech_model_from_sim_result,
        ml_model_from_sim_result,
    )


@pytest.fixture
def hybrid_model_mm_no_node_9(mech_model_from_sim_result, ml_model_from_sim_result):
    mech_model_from_sim_result = deepcopy(mech_model_from_sim_result)
    mech_model_from_sim_result.select_genes([f"Node_{i}" for i in range(9)])
    return HybridModel(mech_model_from_sim_result, ml_model_from_sim_result)


@pytest.fixture
def trainers_for_all_models(sim_result, hybrid_model_mm_no_node_9) -> list[Trainer]:
    trainers: list[Trainer] = []
    for model_class, model_args in zip(
        [CellBox, PhoenixOnlyAddLayers, PhoenixAddAndProdLayers],
        [
            {"self_activations": False},
            {"hidden_size": 2, "integration_method": "rk4"},
            {"hidden_size": 2, "integration_method": "rk4"},
        ],
    ):
        expr_data = sim_result.get_no_noise_ko_timeseries_expressiondata()

        model = model_class(
            modelled_gene_labels=expr_data.genes,
            condition_names=expr_data.all_environments,
            **model_args,
        )
        trainer = Trainer(
            model,
            sim_result,
            nr_training_steps=2,
            optimizer_kwargs={"lr": 0.001},
        )
        trainers.append(trainer)

    # Add hybrid model
    trainers.append(
        Trainer(
            hybrid_model_mm_no_node_9,
            sim_result,
            nr_training_steps=2,
            optimizer_kwargs={"lr": 0.001},
        )
    )

    return trainers


@pytest.fixture
def result_list_from_jobs(trainers_for_all_models) -> FittedTrainerList:
    trainers = deepcopy(trainers_for_all_models)
    for trainer in trainers:
        trainer.save_path = (
            trainer.save_path.parent.parent / "expected_output" / trainer.save_path.name
        )
    result_list = FittedTrainerList.from_train_jobs(trainers)
    return result_list
