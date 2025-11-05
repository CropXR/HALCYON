from pathlib import Path
import random
from typing import Tuple
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from halcyon.Datasets.ExperimentCondition import ExperimentCondition
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.Datasets.ExpressionData import ExpressionData
from halcyon.FittedTrainingJobList import FittedTrainingJobList
from halcyon.Models.CellBox import CellBox
from halcyon.Models.NeuralODE import PhoenixAddAndProdLayers
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from scipy.integrate import solve_ivp

from halcyon.Simulators.GNWSimRunner import GNWSimRunner, SimulationResult
from halcyon.Trainer import TrainingJob


@pytest.fixture(scope="session", autouse=True)
def set_random_seed():
    seed = 174
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture
def get_interactions() -> dict:
    return {
        ("G5", "G1"): 0.2,
        ("G1", "G2"): -0.1,
        ("G2", "G3"): 0.2,
        ("G3", "G4"): -0.4,
        ("G4", "G5"): -0.2,
    }


@pytest.fixture
def create_synthetic_gene_network(get_interactions) -> Tuple[RegulatoryNetwork, dict]:
    interactions = get_interactions
    graph = nx.DiGraph()
    graph.add_nodes_from(
        set(g for pair in interactions.keys() for g in pair)
    )  # Extract unique genes

    for (from_gene, to_gene), influence in interactions.items():
        graph.add_edge(from_gene, to_gene, influence=influence)

    return RegulatoryNetwork(graph), interactions


@pytest.fixture
def create_synthetic_expressions(get_interactions) -> ExpressionDataWithConditions:
    interactions = get_interactions
    initial_state = [0.1, 0.2, 0.3, 0.4, 0.5]
    timespan = [0, 5]
    genes = ("G1", "G2", "G3", "G4", "G5")

    def dy_dt(t, state):
        return [
            interactions.get(("G5", "G1"), 0) * state[4],  # dG1/dt
            interactions.get(("G1", "G2"), 0) * state[0],  # dG2/dt
            interactions.get(("G2", "G3"), 0) * state[1],  # dG3/dt
            interactions.get(("G3", "G4"), 0) * state[2],  # dG4/dt
            interactions.get(("G4", "G5"), 0) * state[3],  # dG5/dt
        ]

    y_out = solve_ivp(dy_dt, timespan, initial_state, method="RK45")
    df = pd.DataFrame(y_out.y, index=genes, columns=y_out.t)
    condition = ExperimentCondition(
        all_genes=genes,
        all_environments=("test1", "test2"),
        ko_gene=None,
        environment=None,
    )
    return ExpressionDataWithConditions({condition: ExpressionData(df)})


@pytest.fixture
def get_synthetic_data(create_synthetic_gene_network, create_synthetic_expressions):
    regulatory_network, interactions = create_synthetic_gene_network
    return regulatory_network, interactions, create_synthetic_expressions


@pytest.fixture
def create_ko_from_sample_data() -> ExpressionDataWithConditions:
    input_path = Path(
        "tests/grnsimdummydata/InSilicoSize10-Yeast1_knockout_timeseries.tsv"
    )
    return ExpressionDataWithConditions.from_genenetweaver_ko_timeseries_output(
        input_path
    )


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
def train_jobs(sim_result):
    train_jobs: list[TrainingJob] = []
    train_steps = 2

    train_jobs.extend(
        [
            TrainingJob(
                PhoenixAddAndProdLayers,
                sim_result,
                nr_training_steps=train_steps,
                model_kwargs={"hidden_size": 2},
                optimizer_kwargs={"lr": 0.1},
            )
        ]
    )

    train_jobs.extend(
        [
            TrainingJob(
                CellBox,
                sim_result,
                nr_training_steps=train_steps,
                model_kwargs={"self_activations": True},
                optimizer_kwargs={"lr": 0.1},
            )
        ]
    )

    for train_job in train_jobs:
        train_job.save_path = (
            train_job.save_path.parent.parent
            / "expected_output"
            / train_job.save_path.name
        )
    return train_jobs


@pytest.fixture
def result_list_from_jobs(train_jobs):
    result_list = FittedTrainingJobList.from_train_jobs(train_jobs)
    return result_list
