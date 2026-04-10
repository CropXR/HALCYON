from pathlib import Path
from copy import deepcopy

from pathos.multiprocessing import ProcessingPool as Pool
import multiprocess as mp
import logging
import random
from itertools import cycle
import os
from functools import partial
import pandas as pd

import mlflow
import torch
import numpy as np
from dotenv import load_dotenv

from halcyon.ModelRunner import ModelRunner, run_training_job
from halcyon.Models.Losses import MSELossFlexible
from halcyon.Models.MechanisticModel import PyTorchODE
from halcyon.Simulators import SimulationResult
from halcyon.Models import HybridModel, PhoenixAddAndProdLayers


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
    mlflow.set_experiment(experiment_id="7")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    # 50 genes with  high noise
    sim_runs_high_noise = mlflow.search_runs(
        experiment_ids=["4"],
        filter_string="tags.run_type = 'dataset' and params.noise_level = 'HIGH' and params.nr_genes = '50'",
    )

    # 50 genes with low noise
    sim_runs_low_noise = mlflow.search_runs(
        experiment_ids=["4"],
        filter_string="tags.run_type = 'dataset' and params.noise_level = 'LOW' and params.nr_genes = '50'",
    )

    sim_runs = pd.concat([sim_runs_high_noise, sim_runs_low_noise])

    for sim_run_id in sim_runs["sim_run_id"]:
        train_hybrid_model_on(sim_run_id)


def train_hybrid_model_on(sim_run_id: str):
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
    devices = cycle(["cuda:1", "cuda:0"])
    batch_size = 32
    optimizer_kwargs = {"lr": 0.01}

    nr_epochs = 2000

    model_jobs: list[ModelRunner] = []

    ml_model = PhoenixAddAndProdLayers(
        modelled_gene_labels=single_ko_dataset.genes,
        hidden_size=10,
        condition_names=single_ko_dataset.all_environments,
        synthesis_only_positive=False,
        device=device,
    )

    model_runner = ModelRunner(
        model=ml_model,
        batch_size=batch_size,
        optimizer_kwargs=optimizer_kwargs,
        loss=loss,
    )

    model_jobs.append(model_runner)

    for genes_to_select in sim_output.network._clustering:
        device = next(devices)
        mm = PyTorchODE.from_sbml(
            sim_output.data_path / "my_network.xml",
            condition_names=[],
            from_gnw=True,
            fill_value_for_gene_not_in_model=0.5,
            device=device,
        )

        # Select only genes from specific module
        mm.select_genes(genes_to_select)
        mm.set_timescaler(time_scaling_factor)

        loss = MSELossFlexible()
        # These are selected based on best results in experiment_id 4
        ml_model = PhoenixAddAndProdLayers(
            modelled_gene_labels=single_ko_dataset.genes,
            hidden_size=10,
            condition_names=single_ko_dataset.all_environments,
            synthesis_only_positive=False,
            device=device,
        )

        hybrid_model = HybridModel(deepcopy(mm), deepcopy(ml_model), device=device)

        model_runner = ModelRunner(
            model=hybrid_model,
            batch_size=batch_size,
            optimizer_kwargs=optimizer_kwargs,
            loss=loss,
        )

        model_jobs.append(model_runner)

    nr_processes = 16
    with Pool(nodes=nr_processes) as pool:
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
            model_jobs,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
