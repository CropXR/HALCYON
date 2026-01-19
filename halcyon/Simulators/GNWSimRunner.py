import logging
from pathlib import Path
import re
import shutil
import subprocess
from dataclasses import dataclass, fields
import networkx as nx
import matplotlib.pyplot as plt

from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.RegulatoryNetwork import RegulatoryNetwork


@dataclass
class SimulationResult:
    """Container for storing the results of a simulated gene network.

    Attributes:
        data_path (Path): Directory for output files.
        network_name (str): File name to use for outputting the files.
        nr_genes (int): Number of genes in the network.
        nr_modules (int): Number of modules in the network.
        sparsity (float): How sparse the network is
            (higher values mean fewer connections).
        keep_self_interactions (bool): Whether self-interactions are kept.
        network (RegulatoryNetwork): The generated gene regulatory network.
    """

    data_path: Path
    network_name: str
    nr_genes: int
    nr_modules: int
    sparsity: float
    keep_self_interactions: bool
    network: RegulatoryNetwork

    def items(self):
        """Iterate over field names and their values (as if this were a dict)"""
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def double_ko_timeseries_nonoise_file_path(self) -> Path:
        """Get path to double knockout timeseries file."""
        out_path = (
            self.data_path / f"{self.network_name}_nonoise_dualknockout_timeseries0.tsv"
        )
        assert out_path.exists()
        return out_path

    def get_double_ko_expressiondata(self) -> ExpressionDataWithConditions:
        double_ko_expressions = ExpressionDataWithConditions.from_gnw_double_ko_files(
            expression_file_path=self.double_ko_timeseries_nonoise_file_path(),
            ko_file_path=self.double_ko_index_file_path(),
        )
        return double_ko_expressions

    def double_ko_index_file_path(self) -> Path:
        """Get path to double knockout index file."""
        out_path = self.data_path / f"{self.network_name}_dualknockouts_indexes.tsv"
        assert out_path.exists()
        return out_path

    def get_no_noise_ko_timeseries_expressiondata(self) -> ExpressionDataWithConditions:
        """Get knockout timeseries without noise data."""
        out_path = (
            self.data_path / f"{self.network_name}_nonoise_knockout_timeseries0.tsv"
        )
        assert out_path.exists()

        return ExpressionDataWithConditions.from_genenetweaver_ko_timeseries_output(
            out_path
        )


class GNWSimRunner:
    def __init__(self, settings_path: Path, jar_path=None):
        """Create GeneNetweaver simulation runner. Point to settings file for settings, and jar file for running the simulations (default = '../../GRNSimulator/CropXRSimulator.jar' relative to this file)"""
        self.settings_path = settings_path
        if jar_path is None:
            # Default jar path relative to this file
            self.jar_path = (
                Path(__file__).parents[2] / "GRNSimulator" / "CropXRSimulator.jar"
            )
        else:
            self.jar_path = jar_path
        assert self.jar_path.exists(), f"Jar file {self.jar_path} does not exist"
        assert (
            self.settings_path.exists()
        ), f"Settings file {self.settings_path} does not exist"
        self.start_of_arg = (
            "java",
            "-jar",
            str(self.jar_path),
            "-c",
            str(self.settings_path),
        )

    def run_simulation(
        self,
        n_genes: int,
        sparsity: float,
        num_modules: int,
        network_name: str,
        output_path: Path,
        keep_self_interactions: bool = True,
        network_type: int = 1,
    ) -> SimulationResult:
        """Perform a genenetweaver network generation + simulation run.

        Args:
            n_genes (int): Number of genes in the network
            sparsity (float): How sparse the network should be (higher values will result in fewer connections)
            num_modules (int): Modularity; number of modules in the network.
            network_name (str): File name to use for outputting the files
            output_path (Path): Directory where files should be stored.
            network_type (int): Specifies the type of network to create (0=Random,
                1=Modular-Scale-Free). (default: 1)
            keep_self_interactions (bool, optional): _description_. Defaults to True.
        """
        assert network_type in [0, 1], "network_type must be 0 or 1"
        # Step 1: generate network
        logging.info("Generating network")
        output_path.mkdir(parents=True, exist_ok=True)
        step_1_args = list(self.start_of_arg) + [
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
        step_2_args = list(self.start_of_arg) + [
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
        )
