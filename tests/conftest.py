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
            {"self_activations": True},
            {"hidden_size": 2},
            {"hidden_size": 2},
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
            optimizer_kwargs={"lr": 0.1},
        )
        trainers.append(trainer)

    # # Add hybrid model
    # # TODO remove this hotfix
    # #  But for now keeping it out
    # trainers.append(
    #     Trainer(
    #         hybrid_model_mm_no_node_9,
    #         sim_result,
    #         nr_training_steps=2,
    #         optimizer_kwargs={"lr": 0.1},
    #     )
    # )

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
