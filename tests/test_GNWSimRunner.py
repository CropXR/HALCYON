from pathlib import Path
from halcyon.Simulators.GNWSimRunner import GNWSimRunner

import hashlib


def create_file_hash(path: Path):
    """Used to check if file contents are indentical quickly"""
    hasher = hashlib.md5()
    with path.open("rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def test_gnw_generation():
    gnw_sim = GNWSimRunner(
        settings_path=Path("setting_files/gnw_testing_runs_settings.txt")
    )
    generated_data_path = Path("tests/generated_test_data")
    expected_data_path = Path("tests/expected_output")
    gnw_sim.run_simulation(
        n_genes=10,
        sparsity=0.1,
        num_modules=2,
        network_name="integration_test_network",
        output_path=generated_data_path,
    )
    gt_files = expected_data_path.iterdir()
    generated_files = generated_data_path.iterdir()
    for gt_file, expected_file in zip(gt_files, generated_files):
        assert create_file_hash(gt_file) == create_file_hash(
            expected_file
        ), f"File {gt_file} does not match expected output"
