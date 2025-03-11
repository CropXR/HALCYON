from typing import Tuple
import networkx as nx
import pandas as pd
from mlmmint.ExpressionData import ExpressionData
from mlmmint.GeneNetwork import GeneNetwork
from scipy.integrate import solve_ivp


def get_interactions() -> dict:
    return {
        ("G5", "G1"): 0.2,
        ("G1", "G2"): -0.1,
        ("G2", "G3"): 0.2,
        ("G3", "G4"): -0.4,
        ("G4", "G5"): -0.2,
    }


# Create and return the gene network as a GeneNetwork object
def create_synthetic_gene_network() -> Tuple[GeneNetwork, dict]:
    interactions = get_interactions()
    graph = nx.DiGraph()
    graph.add_nodes_from(
        set(g for pair in interactions.keys() for g in pair)
    )  # Extract unique genes

    for (from_gene, to_gene), influence in interactions.items():
        graph.add_edge(from_gene, to_gene, influence=influence)

    return GeneNetwork(graph), interactions


# Create synthetic gene expression time series using ODEs
def create_synthetic_expressions() -> ExpressionData:
    interactions = get_interactions()
    initial_state = [0.1, 0.2, 0.3, 0.4, 0.5]
    timespan = [0, 5]

    def dy_dt(t, state):
        return [
            interactions.get(("G5", "G1"), 0) * state[4],  # dG1/dt
            interactions.get(("G1", "G2"), 0) * state[0],  # dG2/dt
            interactions.get(("G2", "G3"), 0) * state[1],  # dG3/dt
            interactions.get(("G3", "G4"), 0) * state[2],  # dG4/dt
            interactions.get(("G4", "G5"), 0) * state[3],  # dG5/dt
        ]

    y_out = solve_ivp(dy_dt, timespan, initial_state, method="RK45")
    df = pd.DataFrame(y_out.y, index=["G1", "G2", "G3", "G4", "G5"], columns=y_out.t)
    return ExpressionData(df)
