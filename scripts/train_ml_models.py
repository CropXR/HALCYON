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
from halcyon.Simulators import SimulationResult
from halcyon.Models import PhoenixAddAndProdLayers


def main(
    sim_run_id: Annotated[
        str, typer.Option(help="The MLFlow simulation run ID to train on.")
    ],
    devices: Annotated[
        list[str],
        typer.Option(
            "--device",
            help="List of devices. Can specify multiple: E.g. --device cuda:0 --device cuda:1 .",
        ),
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size.")],
    nr_processes: Annotated[
        int, typer.Option(help="Number of processes to run in parallel.")
    ],
    nr_epochs: Annotated[int, typer.Option(help="Number of epochs.")],
    hidden_size_list: Annotated[
        list[int],
        typer.Option(
            "--hidden-size",
            help="List of hidden sizes to use for the ML model. Can specify multiple: E.g. --hidden-size 10 --hidden-size 20 --hidden-size 100 and will run them in parallel",
        ),
    ],
    do_only_pos_list: Annotated[
        list[bool],
        typer.Option(
            "--do-only-pos",
            help="True or false for whether to only allow positive synthesis in the ML model.",
        ),
    ],
    optimizer_lr: Annotated[float, typer.Option(help="Learning rate for optimizer.")],
):
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
    experiment_name = "Halcyon sandbox experiment"
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tag("Tool name", "HALCYON")

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
    device = cycle(devices)
    optimizer_kwargs = {"lr": optimizer_lr}
    train_jobs: list[ModelRunner] = []

    # Data-driven models
    for hidden_size in hidden_size_list:
        for do_only_pos in do_only_pos_list:
            ml_model = PhoenixAddAndProdLayers(
                modelled_gene_labels=single_ko_dataset.genes,
                hidden_size=hidden_size,
                condition_names=single_ko_dataset.all_environments,
                synthesis_only_positive=do_only_pos,
                device=next(device),
            )

            model_runner = ModelRunner(
                model=ml_model,
                batch_size=batch_size,
                optimizer_kwargs=optimizer_kwargs,
            )
            train_jobs.append(model_runner)

    for job in train_jobs:
        logging.info(f"{job=}")

    # Do the training
    mp.set_start_method("spawn", force=True)

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
    typer.run(main)
