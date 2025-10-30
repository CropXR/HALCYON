from dataclasses import asdict
from pathlib import Path

import pytest
from halcyon.Simulators.GNWSimRunner import GNWSimRunner, SimulationResult

import hashlib


def create_file_hash(path: Path):
    """Used to check if file contents are indentical quickly"""
    hasher = hashlib.md5()
    with path.open("rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


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


def test_gnw_reproducible(sim_result):
    expected_data_path = Path("tests/expected_output")
    generated_files = sim_result.data_path.iterdir()

    for generated_file in generated_files:
        expected_file = expected_data_path / generated_file.name
        assert create_file_hash(generated_file) == create_file_hash(
            expected_file
        ), f"File {generated_file} does not match hash of {expected_file}"

    expected_sim_result = SimulationResult(
        data_path=Path("tests/generated_test_data").resolve(),
        network_name="integration_test_network",
        nr_genes=10,
        nr_modules=2,
        sparsity=0.1,
        keep_self_interactions=True,
    )
    assert asdict(sim_result) == asdict(expected_sim_result)
