from unittest.mock import MagicMock

import numpy as np
import pytest
from halcyon.Datasets.ExperimentCondition import ExperimentCondition
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.Evaluator import Evaluator
from halcyon.Models.CellBox import CellBox
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Datasets.ExpressionData import ExpressionData

import networkx as nx
from scipy.integrate import solve_ivp
import pandas as pd


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
def synthetic_gene_network(get_interactions) -> tuple[RegulatoryNetwork, dict]:
    interactions = get_interactions
    graph = nx.DiGraph()
    graph.add_nodes_from(
        set(g for pair in interactions.keys() for g in pair)
    )  # Extract unique genes

    for (from_gene, to_gene), influence in interactions.items():
        graph.add_edge(from_gene, to_gene, influence=influence)

    return RegulatoryNetwork(graph)


@pytest.fixture
def synthetic_expressions(get_interactions) -> ExpressionDataWithConditions:
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


def test_compare_inferred_network(synthetic_gene_network, synthetic_expressions):
    my_evaluator = Evaluator(synthetic_gene_network, synthetic_expressions)
    mock_model = MagicMock()
    mock_model.get_inferred_network.return_value = synthetic_gene_network
    # Edge jaccard score
    comparison_output = my_evaluator.compare_edge_jaccard_gt_vs_inferred_network(
        mock_model
    )
    assert comparison_output == 1

    # Empty network
    mock_model.get_inferred_network.return_value = RegulatoryNetwork(nx.DiGraph())
    comparison_output = my_evaluator.compare_edge_jaccard_gt_vs_inferred_network(
        mock_model
    )
    assert comparison_output == 0


def test_roc_auc_inferred_network(synthetic_gene_network, synthetic_expressions):

    my_evaluator = Evaluator(synthetic_gene_network, synthetic_expressions)
    cellbox = CellBox(synthetic_gene_network.node_names, [], self_activations=False)
    cellbox._set_weight_matrix_to_ones()
    comparison_output = my_evaluator.auroc_score_edge_detection(cellbox)
    assert np.allclose(comparison_output, 0.625)
