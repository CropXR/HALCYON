from copy import deepcopy
from pathlib import Path

from matplotlib import pyplot as plt
import logging
import random
from itertools import cycle
import os

import mlflow
import torch
import numpy as np
from dotenv import load_dotenv

from halcyon.ModelRunner import ModelRunner
from halcyon.Models import PyTorchODE
from halcyon.Simulators import SimulationResult


def main():
    # Do setup
    seed = 174
    # for CPU
    torch.manual_seed(seed)
    # for all GPUs
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    load_dotenv()

    logging.info(f'Tracking to {os.getenv("MLFLOW_TRACKING_URI")}')
    mlflow.set_experiment(experiment_id="6")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    # Run on 20, 50, 100 with medium noise
    sim_run_ids = mlflow.search_runs(
        experiment_ids=["4"],
        filter_string="tags.run_type = 'dataset' and params.noise_level = 'MEDIUM'",
    )

    for sim_run_id in sim_run_ids["run_id"]:
        run = mlflow.get_run(sim_run_id)
        params = run.data.params
        logging.info(f"Loaded run with params: {params}")

        # Load data
        ml_flow_download_path = mlflow.artifacts.download_artifacts(
            run_id=sim_run_id, artifact_path="dataset"
        )
        sim_output = SimulationResult.load_from_pkl(
            Path(ml_flow_download_path) / "simulation_result.pkl"
        )
        sim_output.network.do_clustering()

        # From sim output to datasets
        single_ko_dataset, double_ko_dataset, time_scaling_factor = (
            sim_output.create_datasets()
        )

        # No need to split off single KO for validation
        train_dataset_name = "train_single_ko"
        test_dataset_name = "double_ko"
        dataset_dict = {
            train_dataset_name: single_ko_dataset,
            test_dataset_name: double_ko_dataset,
        }

        # Create train jobs
        device = cycle(["cuda:0", "cuda:1"])
        batch_size = 32

        mm_jobs: list[ModelRunner] = []

        for genes_to_select in sim_output.network._clustering:
            mm = PyTorchODE.from_sbml(
                sim_output.data_path / "my_network.xml",
                condition_names=[],
                from_gnw=True,
                device=next(device),
                fill_value_for_gene_not_in_model=0.5,
            )

            # Select only genes from specific module
            mm.select_genes(genes_to_select)
            mm.set_timescaler(time_scaling_factor)

            mm_trainer = ModelRunner(
                mm,
                batch_size=batch_size,
            )
            mm_jobs.append(mm_trainer)

        for train_job in mm_jobs:
            train_job.train_and_predict_for_mlflow(
                dataset_dict=dataset_dict,
                simulation_parent_id=sim_run_id,
                nr_epochs=0,
                train_data_key_in_dict=train_dataset_name,
                test_data_key_in_dict=test_dataset_name,
            )


if __name__ == "__main__":
    main()
