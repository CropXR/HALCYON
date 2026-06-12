from pathlib import Path
from copy import deepcopy

# import multiprocessing as mp
import torch.multiprocessing as mp
import logging
import random
from itertools import cycle
import os
from functools import partial

from pathos.multiprocessing import ProcessingPool as Pool

import mlflow
import torch
import numpy as np
from dotenv import load_dotenv

from halcyon.ModelRunner import ModelRunner, run_training_job
from halcyon.ModelRunner.run_training_job import prepare_dataset
from halcyon.Models import HybridModel, PyTorchODE, PhoenixAddAndProdLayers, CellBox


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

    mlflow.set_experiment("Hybrid but with different ML")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    sim_run_id = "f02bf732b2ba4817b254bd5174f1549b"
    devices = cycle(["cuda:1", "cuda:0"])
    lr = 0.01
    nr_processes = 16
    nr_epochs = 2000
    batch_size = 32
    (
        sim_output,
        single_ko_dataset,
        time_scaling_factor,
        train_dataset_name,
        val_dataset_name,
        test_dataset_name,
        dataset_dict,
    ) = prepare_dataset(sim_run_id)

    # Create train jobs
    optimizer_kwargs = {"lr": lr}

    model_jobs: list[ModelRunner] = []

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

        for ml_model in [
            phoenix_factory(single_ko_dataset, device),
            cellbox_factory(single_ko_dataset, device),
        ]:
            hybrid_model = HybridModel(deepcopy(mm), deepcopy(ml_model), device=device)

            model_runner = ModelRunner(
                model=hybrid_model,
                batch_size=batch_size,
                optimizer_kwargs=optimizer_kwargs,
            )

            model_jobs.append(model_runner)

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


def phoenix_factory(single_ko_dataset, device):
    return PhoenixAddAndProdLayers(
        modelled_gene_labels=single_ko_dataset.genes,
        hidden_size=50,
        condition_names=single_ko_dataset.all_environments,
        synthesis_only_positive=True,
        device=device,
    )


def cellbox_factory(single_ko_dataset, device):
    return CellBox(
        modelled_gene_labels=single_ko_dataset.genes,
        condition_names=single_ko_dataset.all_environments,
        device=device,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
