from copy import deepcopy
from pathlib import Path
import random
from typing import Tuple
import networkx as nx

import numpy as np

import pytest
import torch

from halcyon.Simulators.run_genenetweaver import run_gnw, NoiseLevel, SimulationResult
from halcyon.Datasets import ExpressionDataWithConditions
from halcyon.Models import (
    MLP,
    CellBox,
    PhoenixAddAndProdLayers,
    PhoenixOnlyAddLayers,
    HybridModel,
    PyTorchODE,
)
from halcyon.ModelRunner import ModelRunner


from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def get_env_variables():
    load_dotenv()


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
            "tests/expected_output/integration_test_network_nonoise_knockout_timeseries0.tsv"
        )
    )
    return expressions


@pytest.fixture
def sim_result() -> SimulationResult:
    generated_data_path = Path("tests/generated_test_data")
    sim_result = run_gnw(
        settings_dir=Path("setting_files"),
        n_genes=10,
        sparsity=0.1,
        num_modules=2,
        network_name="integration_test_network",
        output_path=generated_data_path,
        noiseless_for_testing=True,
        noise_level=NoiseLevel.MEDIUM,
    )
    return sim_result


@pytest.fixture
def sim_result_lower_noise() -> SimulationResult:
    generated_data_path = Path("tests/generated_test_data_lower_noise")
    sim_result = run_gnw(
        settings_dir=Path("setting_files"),
        n_genes=10,
        sparsity=0.1,
        num_modules=2,
        network_name="integration_test_network",
        output_path=generated_data_path,
        noiseless_for_testing=True,
        noise_level=NoiseLevel.LOW,
    )
    return sim_result


@pytest.fixture
def expr_data_from_sim(sim_result: SimulationResult) -> ExpressionDataWithConditions:
    return sim_result.get_single_ko_timeseries_expressiondata()


@pytest.fixture
def normalised_expr_data_from_sim(
    sim_result: SimulationResult,
) -> ExpressionDataWithConditions:
    expr_data = deepcopy(sim_result.get_single_ko_timeseries_expressiondata())
    sim_result._time_scale_all_datasets_and_get_scaling_factor([expr_data])
    return expr_data


@pytest.fixture
def mech_model_from_sim_result(
    sim_result: SimulationResult, expr_data_from_sim: ExpressionDataWithConditions
) -> PyTorchODE:
    model = PyTorchODE.from_sbml(
        sim_result.data_path / "integration_test_network.xml",
        condition_names=expr_data_from_sim.all_environments,
        from_gnw=True,
    )
    return model


@pytest.fixture
def ml_model_from_sim_result(
    expr_data_from_sim: ExpressionDataWithConditions,
) -> PhoenixAddAndProdLayers:
    model = PhoenixAddAndProdLayers(
        expr_data_from_sim.genes, expr_data_from_sim.all_environments, hidden_size=2
    )
    return deepcopy(model)


@pytest.fixture
def hybrid_model(
    mech_model_from_sim_result: PyTorchODE,
    ml_model_from_sim_result: PhoenixAddAndProdLayers,
) -> HybridModel:
    return HybridModel(mech_model_from_sim_result, ml_model_from_sim_result)


@pytest.fixture
def hybrid_model_with_zero_ml_influence(
    mech_model_from_sim_result: PyTorchODE,
    ml_model_from_sim_result: PhoenixAddAndProdLayers,
) -> HybridModel:
    ml_model_from_sim_result = deepcopy(ml_model_from_sim_result)
    ml_model_from_sim_result._set_raw_per_gene_importance(-50)
    return HybridModel(
        mech_model_from_sim_result,
        ml_model_from_sim_result,
    )


@pytest.fixture
def hybrid_model_mm_no_node_9(
    mech_model_from_sim_result: PyTorchODE,
    ml_model_from_sim_result: PhoenixAddAndProdLayers,
) -> HybridModel:
    mech_model_from_sim_result = deepcopy(mech_model_from_sim_result)
    mech_model_from_sim_result.select_genes([f"Node_{i}" for i in range(9)])
    return HybridModel(
        deepcopy(mech_model_from_sim_result),
        deepcopy(ml_model_from_sim_result),
    )


@pytest.fixture
def trainers_for_all_models(
    sim_result: SimulationResult, hybrid_model_mm_no_node_9: HybridModel
) -> list[ModelRunner]:
    trainers: list[ModelRunner] = []
    expr_data = sim_result.get_single_ko_timeseries_expressiondata()

    time_scaler = sim_result._time_scale_all_datasets_and_get_scaling_factor(
        [expr_data]
    )

    model_configs = [
        (PhoenixOnlyAddLayers, {"hidden_size": 2}),
        (PhoenixAddAndProdLayers, {"hidden_size": 2}),
        (MLP, {"hidden_size": 2, "nr_layers": 2}),
        (CellBox, {"self_activations": False}),
    ]

    for model_class, model_args in model_configs:
        model = model_class(
            modelled_gene_labels=expr_data.genes,
            condition_names=expr_data.all_environments,
            **model_args,
        )
        trainer = ModelRunner(
            model,
            optimizer_kwargs={"lr": 0.1},
        )
        trainers.append(trainer)
    hybrid_model_mm_no_node_9 = deepcopy(hybrid_model_mm_no_node_9)
    hybrid_model_mm_no_node_9._mm.set_timescaler(time_scaler)
    # Add hybrid model
    trainers.append(
        ModelRunner(
            hybrid_model_mm_no_node_9,
            optimizer_kwargs={"lr": 0.1},
        )
    )

    return trainers


@pytest.fixture(params=["phoenix", "mlp", "cellbox"])
def ode_model(request):
    gene_labels = ["a", "b", "c"]
    if request.param == "phoenix":
        return PhoenixOnlyAddLayers(gene_labels, ("test_cond",), hidden_size=2)
    elif request.param == "mlp":
        return MLP(
            gene_labels,
            ("test_cond",),
            hidden_size=2,
            nr_layers=2,
            synthesis_only_positive=True,
        )
    elif request.param == "cellbox":
        return CellBox(gene_labels, ("test_cond",))


@pytest.fixture
def expected_batch():
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
                    3.7892e-01,
                    2.2609e-01,
                    1.5060e-01,
                    1.1168e-01,
                    9.1488e-02,
                    8.1063e-02,
                    7.5716e-02,
                    7.2985e-02,
                    7.1594e-02,
                    7.0887e-02,
                    7.0527e-02,
                    7.0345e-02,
                    7.0252e-02,
                    7.0205e-02,
                    7.0181e-02,
                    7.0169e-02,
                    7.0163e-02,
                    7.0159e-02,
                    7.0158e-02,
                    7.0157e-02,
                ],
                [
                    6.3392e-01,
                    7.8251e-02,
                    9.6593e-03,
                    1.1923e-03,
                    1.4718e-04,
                    1.8169e-05,
                    2.2429e-06,
                    2.7693e-07,
                    3.4187e-08,
                    4.2255e-09,
                    5.7365e-10,
                    1.3770e-10,
                    2.9949e-11,
                    8.4016e-12,
                    3.0030e-12,
                    9.4969e-13,
                    2.8660e-13,
                    6.7683e-14,
                    1.7352e-14,
                    4.9147e-15,
                    1.2522e-15,
                ],
                [
                    9.4803e-01,
                    6.7610e-01,
                    4.9957e-01,
                    3.7960e-01,
                    3.2036e-01,
                    2.9564e-01,
                    2.8572e-01,
                    2.8161e-01,
                    2.7979e-01,
                    2.7895e-01,
                    2.7854e-01,
                    2.7834e-01,
                    2.7824e-01,
                    2.7819e-01,
                    2.7816e-01,
                    2.7815e-01,
                    2.7814e-01,
                    2.7814e-01,
                    2.7814e-01,
                    2.7814e-01,
                    2.7814e-01,
                ],
                [
                    7.4355e-01,
                    7.1641e-01,
                    7.2545e-01,
                    7.2801e-01,
                    7.2853e-01,
                    7.2862e-01,
                    7.2864e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                ],
                [
                    1.3590e-01,
                    6.2013e-01,
                    6.8386e-01,
                    6.8558e-01,
                    6.8341e-01,
                    6.8222e-01,
                    6.8171e-01,
                    6.8151e-01,
                    6.8143e-01,
                    6.8140e-01,
                    6.8138e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                ],
                [
                    9.9317e-01,
                    9.3461e-01,
                    9.0604e-01,
                    8.9690e-01,
                    8.9407e-01,
                    8.9320e-01,
                    8.9293e-01,
                    8.9284e-01,
                    8.9282e-01,
                    8.9281e-01,
                    8.9280e-01,
                    8.9280e-01,
                    8.9280e-01,
                    8.9280e-01,
                    8.9280e-01,
                    8.9280e-01,
                    8.9280e-01,
                    8.9280e-01,
                    8.9280e-01,
                    8.9280e-01,
                    8.9280e-01,
                ],
                [
                    1.1174e-02,
                    5.5135e-01,
                    5.6163e-01,
                    5.6223e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                    5.6226e-01,
                ],
                [
                    2.3878e-01,
                    1.0041e-01,
                    3.3816e-02,
                    1.7218e-02,
                    1.3461e-02,
                    1.2628e-02,
                    1.2442e-02,
                    1.2399e-02,
                    1.2388e-02,
                    1.2385e-02,
                    1.2384e-02,
                    1.2384e-02,
                    1.2384e-02,
                    1.2384e-02,
                    1.2384e-02,
                    1.2384e-02,
                    1.2384e-02,
                    1.2384e-02,
                    1.2384e-02,
                    1.2384e-02,
                    1.2384e-02,
                ],
                [
                    2.9734e-01,
                    7.3142e-01,
                    9.2472e-01,
                    9.8041e-01,
                    9.9495e-01,
                    9.9867e-01,
                    9.9962e-01,
                    9.9986e-01,
                    9.9993e-01,
                    9.9995e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                ],
                [
                    9.9509e-01,
                    3.1457e-01,
                    6.2771e-01,
                    9.0034e-01,
                    9.7509e-01,
                    9.9380e-01,
                    9.9846e-01,
                    9.9962e-01,
                    9.9990e-01,
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
                ],
            ],
            [
                [
                    7.2333e-01,
                    3.7890e-01,
                    2.2603e-01,
                    1.5054e-01,
                    1.1165e-01,
                    9.1484e-02,
                    8.1083e-02,
                    7.5752e-02,
                    7.3031e-02,
                    7.1646e-02,
                    7.0942e-02,
                    7.0584e-02,
                    7.0402e-02,
                    7.0310e-02,
                    7.0263e-02,
                    7.0239e-02,
                    7.0227e-02,
                    7.0221e-02,
                    7.0218e-02,
                    7.0216e-02,
                    7.0215e-02,
                ],
                [
                    6.3392e-01,
                    8.7008e-01,
                    8.0575e-01,
                    7.5826e-01,
                    7.4267e-01,
                    7.3823e-01,
                    7.3688e-01,
                    7.3641e-01,
                    7.3622e-01,
                    7.3614e-01,
                    7.3610e-01,
                    7.3608e-01,
                    7.3607e-01,
                    7.3606e-01,
                    7.3606e-01,
                    7.3606e-01,
                    7.3606e-01,
                    7.3606e-01,
                    7.3606e-01,
                    7.3606e-01,
                    7.3606e-01,
                ],
                [
                    9.4803e-01,
                    6.2214e-01,
                    4.0737e-01,
                    2.6901e-01,
                    2.0222e-01,
                    1.7681e-01,
                    1.6818e-01,
                    1.6516e-01,
                    1.6399e-01,
                    1.6349e-01,
                    1.6325e-01,
                    1.6314e-01,
                    1.6308e-01,
                    1.6306e-01,
                    1.6304e-01,
                    1.6303e-01,
                    1.6303e-01,
                    1.6303e-01,
                    1.6303e-01,
                    1.6303e-01,
                    1.6303e-01,
                ],
                [
                    7.4355e-01,
                    7.1677e-01,
                    7.2572e-01,
                    7.2811e-01,
                    7.2856e-01,
                    7.2864e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                    7.2865e-01,
                ],
                [
                    1.3590e-01,
                    6.2013e-01,
                    6.8385e-01,
                    6.8557e-01,
                    6.8340e-01,
                    6.8222e-01,
                    6.8171e-01,
                    6.8151e-01,
                    6.8143e-01,
                    6.8140e-01,
                    6.8138e-01,
                    6.8138e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                    6.8137e-01,
                ],
                [
                    9.9317e-01,
                    9.7242e-01,
                    9.6651e-01,
                    9.6232e-01,
                    9.5992e-01,
                    9.5885e-01,
                    9.5842e-01,
                    9.5825e-01,
                    9.5819e-01,
                    9.5817e-01,
                    9.5816e-01,
                    9.5815e-01,
                    9.5815e-01,
                    9.5815e-01,
                    9.5815e-01,
                    9.5815e-01,
                    9.5815e-01,
                    9.5815e-01,
                    9.5815e-01,
                    9.5815e-01,
                    9.5815e-01,
                ],
                [
                    1.1174e-02,
                    9.0738e-01,
                    9.5806e-01,
                    9.4966e-01,
                    9.4362e-01,
                    9.4149e-01,
                    9.4082e-01,
                    9.4060e-01,
                    9.4052e-01,
                    9.4049e-01,
                    9.4048e-01,
                    9.4047e-01,
                    9.4046e-01,
                    9.4046e-01,
                    9.4046e-01,
                    9.4046e-01,
                    9.4046e-01,
                    9.4046e-01,
                    9.4046e-01,
                    9.4046e-01,
                    9.4046e-01,
                ],
                [
                    2.3878e-01,
                    4.9506e-02,
                    1.0264e-02,
                    2.1280e-03,
                    4.4119e-04,
                    9.1470e-05,
                    1.8965e-05,
                    3.9320e-06,
                    8.1519e-07,
                    1.6904e-07,
                    3.5055e-08,
                    7.1363e-09,
                    1.4638e-09,
                    3.7151e-10,
                    1.1119e-10,
                    4.4989e-11,
                    1.3784e-11,
                    3.5293e-12,
                    1.2534e-12,
                    5.4781e-13,
                    1.5744e-13,
                ],
                [
                    2.9734e-01,
                    7.3402e-01,
                    9.2555e-01,
                    9.8062e-01,
                    9.9501e-01,
                    9.9868e-01,
                    9.9962e-01,
                    9.9987e-01,
                    9.9993e-01,
                    9.9995e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                    9.9996e-01,
                ],
                [
                    9.9509e-01,
                    2.5420e-01,
                    6.9851e-02,
                    2.4340e-02,
                    1.3188e-02,
                    1.0469e-02,
                    9.8085e-03,
                    9.6496e-03,
                    9.6121e-03,
                    9.6035e-03,
                    9.6018e-03,
                    9.6016e-03,
                    9.6016e-03,
                    9.6017e-03,
                    9.6017e-03,
                    9.6017e-03,
                    9.6017e-03,
                    9.6017e-03,
                    9.6017e-03,
                    9.6018e-03,
                    9.6017e-03,
                ],
            ],
        ]
    )
    return experiment_data, expression_data
