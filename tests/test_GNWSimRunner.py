from dataclasses import asdict
from pathlib import Path
import networkx as nx
import json

from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Simulators.NoiseLevel import NoiseLevel

import hashlib

from halcyon.Simulators.SimulationResult import SimulationResult


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

    # Uncomment temporarily to save to json in case new network or smth
    # data = nx.node_link_data(sim_result.network._graph, edges="links")
    # with open(expected_data_path / "expected_network.json", "w") as f:
    #     json.dump(data, f)

    with open(expected_data_path / "expected_network.json") as f:
        expected_network_json = json.load(f)

    expected_network = nx.node_link_graph(expected_network_json, edges="links")
    expected_sim_result = SimulationResult(
        data_path=Path("tests/generated_test_data").resolve(),
        network_name="integration_test_network",
        nr_genes=10,
        nr_modules=2,
        sparsity=0.1,
        network=RegulatoryNetwork(expected_network),
        keep_self_interactions=True,
        noise_level=sim_result.noise_level,
    )
    assert asdict(sim_result) == asdict(expected_sim_result)


def test_plot_gnw_run(sim_result: SimulationResult):
    # Just rough code to see if no errors
    sim_result.network.plot()


def test_sim_result_build_file_path_with_different_noise(sim_result):
    file_name = "knockout_timeseries0.tsv"
    filename_no_noise = sim_result._build_filename(file_name)
    assert (
        filename_no_noise.name.endswith(f"_nonoise_{file_name}")
        and filename_no_noise.exists()
    )

    sim_result.noise_level = NoiseLevel.NOEXPNOISE
    filename_no_exp_noise = sim_result._build_filename(file_name)
    assert (
        filename_no_exp_noise.name.endswith(f"_noexpnoise_{file_name}")
        and filename_no_exp_noise.exists()
    )

    sim_result.noise_level = NoiseLevel.ALLNOISE
    filename_with_noise = sim_result._build_filename(file_name)
    assert (
        filename_with_noise.name.endswith(f"_{file_name}")
        and filename_with_noise.exists()
    )
