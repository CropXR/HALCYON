import logging
from pathlib import Path
import shutil
import subprocess

from halcyon.Datasets.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Simulators.NoiseLevel import NoiseLevel
from halcyon.Simulators.SimulationResult import SimulationResult


def run_gnw(
    settings_dir: Path,
    n_genes: int,
    sparsity: float,
    num_modules: int,
    network_name: str,
    output_path: Path,
    noise_level: NoiseLevel,
    keep_self_interactions: bool = True,
    noiseless_for_testing: bool = False,
    network_type: int = 1,
    grouping_bias: float = 100.0,
    jar_path=None,
) -> SimulationResult:
    """Run a GeneNetWeaver simulation and return the result.

    Args:
        settings_dir: Directory containing the GNW settings file.
        n_genes: Number of genes in the network.
        sparsity: Network sparsity; higher values produce fewer connections.
        num_modules: Number of modules for modular network topology.
        network_name: Base name used for all output files.
        output_path: Directory where simulation output files are written.
        noise_level: Noise level to apply to the simulated expression data.
        keep_self_interactions: Whether to retain self-regulatory interactions. Defaults to True.
        noiseless_for_testing: If True, will return files with no noise (used for testing). Defaults to False.
        network_type: Network topology type. 0 = random, 1 = modular scale-free. Defaults to 1.
        jar_path: Path to the GNW jar file. Defaults to '../../GRNSimulator/CropXRSimulator.jar' relative to this file.
    Returns:
        SimulationResult containing paths to the generated output files and simulation metadata.
    """
    if jar_path is None:
        # Default jar path relative to this file
        jar_path = Path(__file__).parents[2] / "GRNSimulator" / "CropXRSimulator.jar"
    assert jar_path.exists(), f"GNW jar file {jar_path} does not exist"

    settings_path = settings_dir / noise_level.value
    assert settings_path.exists(), f"Settings file {settings_path} does not exist"
    assert network_type in [0, 1], "network_type must be 0 or 1"

    start_of_arg = (
        "java",
        "-jar",
        str(jar_path),
        "-c",
        str(settings_path),
    )

    if noiseless_for_testing:
        logging.warning(
            f"Received {noiseless_for_testing=}. This omits the NoiseLevel that you might have provided."
        )

    # Step 1: generate network
    logging.info("Generating network")
    output_path.mkdir(parents=True, exist_ok=True)
    step_1_args = list(start_of_arg) + [
        "--create",
        f"--network-type={network_type}",
        "--output-net-format=4",
        "--num-genes",
        str(n_genes),
        "--sparsity",
        str(sparsity),
        "--num-modules",
        str(num_modules),
        "--output-path",
        str(output_path),
        "--network-name",
        network_name,
        "--grouping-bias",
        str(grouping_bias),
    ]
    if keep_self_interactions:
        step_1_args.append("--keep-self-interactions")
    if sparsity == 0:
        raise ValueError("Sparsity cannot be zero.")
    step1_process = subprocess.run(
        step_1_args, check=True, capture_output=True, text=True
    )
    logging.info(step1_process.stderr)
    network = RegulatoryNetwork.from_stderr_gnw(
        step1_process.stderr, n_genes, network_name
    )

    # Step 2 run simulation
    logging.info("Simulating network expression over time")
    step_2_args = list(start_of_arg) + [
        "--simulate",
        "--input-net",
        f"{output_path}/{network_name}.xml",
    ]
    step2_process = subprocess.run(
        step_2_args, check=True, capture_output=True, text=True
    )

    # Step 3 -> move the files to the correct directory because simulation step cannot seem to do it
    for line in step2_process.stderr.splitlines():
        if line.startswith("Writing file"):
            file_name = Path(line.split("file ")[1].strip())
            dest = output_path / file_name.name
            shutil.move(str(file_name), str(dest))

    return SimulationResult(
        data_path=output_path.resolve(),
        network_name=network_name,
        nr_genes=n_genes,
        nr_modules=num_modules,
        sparsity=sparsity,
        keep_self_interactions=keep_self_interactions,
        network=network,
        noise_level=noise_level,
        noiseless_for_testing=noiseless_for_testing,
    )
