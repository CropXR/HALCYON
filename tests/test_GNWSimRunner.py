from dataclasses import asdict
from pathlib import Path
import networkx as nx
import json

from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Simulators.GNWSimRunner import SimulationResult

import hashlib


def create_file_hash(path: Path):
    """Used to check if file contents are indentical quickly"""
    hasher = hashlib.md5()
    with path.open("rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def test_gnw_reproducible(sim_result):
    expected_data_path = Path("tests/expected_output")
    generated_files = sim_result.data_path.iterdir()

    for generated_file in generated_files:
        if generated_file.suffix == ".pt":
            # Ignore saved models
            continue
        expected_file = expected_data_path / generated_file.name
        assert create_file_hash(generated_file) == create_file_hash(
            expected_file
        ), f"File {generated_file} does not match hash of {expected_file}"

    expected_network_json = '{"directed": true, "multigraph": false, "graph": {}, "nodes": [{"id": "Node_2"}, {"id": "Node_0"}, {"id": "Node_4"}, {"id": "Node_3"}, {"id": "Node_5"}, {"id": "Node_1"}, {"id": "Node_6"}, {"id": "Node_7"}, {"id": "Node_8"}, {"id": "Node_9"}], "links": [{"color": "red", "source": "Node_2", "target": "Node_0"}, {"color": "green", "source": "Node_2", "target": "Node_2"}, {"color": "green", "source": "Node_2", "target": "Node_3"}, {"color": "red", "source": "Node_0", "target": "Node_0"}, {"color": "green", "source": "Node_0", "target": "Node_2"}, {"color": "green", "source": "Node_0", "target": "Node_3"}, {"color": "green", "source": "Node_0", "target": "Node_4"}, {"color": "red", "source": "Node_0", "target": "Node_7"}, {"color": "green", "source": "Node_0", "target": "Node_8"}, {"color": "red", "source": "Node_4", "target": "Node_0"}, {"color": "green", "source": "Node_4", "target": "Node_2"}, {"color": "red", "source": "Node_4", "target": "Node_4"}, {"color": "red", "source": "Node_3", "target": "Node_0"}, {"color": "green", "source": "Node_3", "target": "Node_2"}, {"color": "green", "source": "Node_3", "target": "Node_3"}, {"color": "red", "source": "Node_3", "target": "Node_4"}, {"color": "red", "source": "Node_5", "target": "Node_1"}, {"color": "red", "source": "Node_5", "target": "Node_5"}, {"color": "red", "source": "Node_1", "target": "Node_1"}, {"color": "red", "source": "Node_1", "target": "Node_5"}, {"color": "green", "source": "Node_1", "target": "Node_6"}, {"color": "green", "source": "Node_1", "target": "Node_9"}, {"color": "green", "source": "Node_6", "target": "Node_6"}, {"color": "green", "source": "Node_7", "target": "Node_8"}]}'
    expected_network = nx.node_link_graph(
        json.loads(expected_network_json), edges="links"
    )
    expected_sim_result = SimulationResult(
        data_path=Path("tests/generated_test_data").resolve(),
        network_name="integration_test_network",
        nr_genes=10,
        nr_modules=2,
        sparsity=0.1,
        network=RegulatoryNetwork(expected_network),
        keep_self_interactions=True,
    )
    assert asdict(sim_result) == asdict(expected_sim_result)


def test_plot_gnw_run(sim_result: SimulationResult):
    # Just rough code to see if no errors
    sim_result.network.plot()
