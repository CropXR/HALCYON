from pathlib import Path
from copy import deepcopy

# import multiprocessing as mp
import torch.multiprocessing as mp
import logging
import random
from itertools import cycle
import os
from functools import partial
from typing import Annotated

import mlflow
import torch
import numpy as np
from dotenv import load_dotenv
import typer

from halcyon.ModelRunner import ModelRunner, run_training_job
from halcyon.Models.Losses import MSELossPlusL1
from halcyon.Simulators import SimulationResult
from halcyon.Models import MLP, CellBox, PhoenixAddAndProdLayers


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
    mlflow.set_experiment(experiment_id="4")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    sim_run_ids = [
        # "d660c1d337c34d98a349605d9b4d5b47",
        "19ba6fc0e8094a66a68f2f67cdca2162",
        "e56b1774f68349209f3b7e672158fa01",
    ]

    for sim_run_id in sim_run_ids:
        train_models_on_simulation_run(sim_run_id)


def train_models_on_simulation_run(sim_run_id: str):
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
    # From sim output to datasets
    single_ko_dataset, double_ko_dataset, time_scaling_factor = (
        sim_output.create_datasets()
    )
    train_ko_dataset, validation_ko_dataset = single_ko_dataset.split_train_test(
        test_fraction=0.25
    )

    train_dataset_name = "train_single_ko"
    val_dataset_name = "val_single_ko"
    test_dataset_name = "double_ko"
    dataset_dict = {
        train_dataset_name: train_ko_dataset,
        val_dataset_name: validation_ko_dataset,
        test_dataset_name: double_ko_dataset,
    }

    # Create train jobs
    device = cycle(["cuda:0", "cuda:1"])
    optimizer_kwargs = {"lr": 0.01}
    batch_size = 32
    train_jobs: list[ModelRunner] = []

    # Phoenix
    for hidden_size in [20, 50, 100]:
        for lambda_value in [0.0, 0.1, 1]:
            for only_pos_synthesis in [True, False]:
                loss = MSELossPlusL1(lambda_param=lambda_value)
                ml_model = PhoenixAddAndProdLayers(
                    modelled_gene_labels=single_ko_dataset.genes,
                    hidden_size=hidden_size,
                    condition_names=single_ko_dataset.all_environments,
                    synthesis_only_positive=only_pos_synthesis,
                    device=next(device),
                )

                model_runner = ModelRunner(
                    model=ml_model,
                    batch_size=batch_size,
                    optimizer_kwargs=optimizer_kwargs,
                    loss=loss,
                )
                train_jobs.append(model_runner)

    # CellBox
    for lambda_value in [0.0, 0.1, 1]:
        loss = MSELossPlusL1(lambda_param=lambda_value)
        cellbox = CellBox(
            modelled_gene_labels=single_ko_dataset.genes,
            condition_names=single_ko_dataset.all_environments,
            device=next(device),
        )

        model_runner = ModelRunner(
            model=cellbox,
            batch_size=batch_size,
            optimizer_kwargs=optimizer_kwargs,
            loss=loss,
        )
        train_jobs.append(model_runner)

    # MLP
    for hidden_size in [10, 20, 50]:
        for nr_layers in [1, 2, 3]:
            for lambda_value in [0.0, 0.1, 1]:
                for only_pos_synthesis in [True, False]:
                    loss = MSELossPlusL1(lambda_param=lambda_value)
                    mlp = MLP(
                        modelled_gene_labels=single_ko_dataset.genes,
                        condition_names=single_ko_dataset.all_environments,
                        hidden_size=hidden_size,
                        nr_layers=nr_layers,
                        device=next(device),
                        synthesis_only_positive=only_pos_synthesis,
                    )

                    model_runner = ModelRunner(
                        model=mlp,
                        batch_size=batch_size,
                        optimizer_kwargs=optimizer_kwargs,
                        loss=loss,
                    )
                    train_jobs.append(model_runner)

    # Do the training
    nr_epochs = 2000
    nr_processes = 16

    with mp.Pool(processes=nr_processes) as pool:
        pool.map(
            partial(
                run_training_job,
                dataset_dict=dataset_dict,
                train_data_key_in_dict=train_dataset_name,
                val_data_key_in_dict=val_dataset_name,
                test_data_key_in_dict=test_dataset_name,
                simulation_parent_id=sim_run_id,
                nr_epochs=nr_epochs,
                do_mlflow_model_logging=False,
            ),
            train_jobs,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
