from pathlib import Path
from copy import deepcopy
import logging
import random
import os
from typing import Annotated

import mlflow
import matplotlib.pyplot as plt
from networkx import number_weakly_connected_components
import torch
import numpy as np
from dotenv import load_dotenv
import typer

from halcyon.Simulators import NoiseLevel, SimulationResult, run_gnw


def main(
    out_path: Annotated[
        Path, typer.Option("--out-path", help="Path to save output files")
    ]
):
    seed = 174
    # for CPU
    torch.manual_seed(seed)
    # for all GPUs
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    load_dotenv()

    mlflow.set_experiment("Halcyon comparing models and datasets")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    nr_genes_list = [20, 50, 100]
    nr_modules_list = [2, 5, 10]
    sparsity_list = [0.1]
    keep_self_interactions = False

    # Assume settings are one directory up and in the setting_files directory
    settings_dir = Path(__file__).parent.parent / "setting_files"

    for nr_genes, nr_modules in zip(nr_genes_list, nr_modules_list):
        for sparsity in sparsity_list:
            for noise in NoiseLevel:
                generated_data_path = (
                    out_path
                    / f"genes_{nr_genes}_modules_{nr_modules}_sparsity_{sparsity}_noise_{noise.name}"
                )
                generated_data_path.mkdir(parents=True, exist_ok=True)

                log_location = generated_data_path / "simulation.log"
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.FileHandler(log_location),
                        logging.StreamHandler(),  # also print to console
                    ],
                    force=True,
                )
                network_name = "my_network"
                sim_output = run_gnw(
                    settings_dir=settings_dir,
                    n_genes=nr_genes,
                    sparsity=sparsity,
                    num_modules=nr_modules,
                    network_name=network_name,
                    output_path=generated_data_path,
                    noise_level=noise,
                    keep_self_interactions=keep_self_interactions,
                )

                sim_output.network.do_clustering()
                assert (
                    number_weakly_connected_components(sim_output.network._graph) == 1
                ), "The generated network is not fully connected."

                sim_output.network.plot()
                fig_path = generated_data_path / "network_plot.png"
                plt.savefig(fig_path)

                sim_output.save_to_pkl(generated_data_path / "simulation_result.pkl")

                with mlflow.start_run(
                    run_name=f"dataset_genes{nr_genes}_sparsity{sparsity}_nr_modules{nr_modules}_noise{noise.name}"
                ):
                    mlflow.set_tag("run_type", "dataset")
                    mlflow.log_params(
                        {
                            "nr_genes": nr_genes,
                            "sparsity": sparsity,
                            "num_modules": nr_modules,
                            "noise_level": noise.name,
                            "seed": seed,
                        }
                    )
                    mlflow.log_artifacts(generated_data_path, artifact_path="dataset")


if __name__ == "__main__":
    typer.run(main)
