from copy import deepcopy

import multiprocess as mp
import random
from itertools import cycle
from functools import partial

import mlflow
import torch
import numpy as np
from dotenv import load_dotenv
from pathos.multiprocessing import ProcessingPool as Pool

from halcyon.ModelRunner.ModelRunner import ModelRunner
from halcyon.ModelRunner.run_training_job import prepare_dataset, run_training_job
from halcyon.Models import best_mlp_factory, HybridModel, PyTorchODE


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

    mlflow.set_experiment("MM with mistakes")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    # 50 genes medium noise
    sim_run_id = "f02bf732b2ba4817b254bd5174f1549b"

    (
        sim_output,
        single_ko_dataset,
        time_scaling_factor,
        train_dataset_name,
        val_dataset_name,
        test_dataset_name,
        dataset_dict,
    ) = prepare_dataset(sim_run_id)

    nr_processes = 16
    nr_epochs = 2000
    batch_size = 32
    lr = 0.01
    devices = cycle(["cuda:1", "cuda:0"])
    optimizer_kwargs = {"lr": lr}

    # Create train jobs
    hybrid_model_jobs: list[ModelRunner] = []
    mm_jobs: list[ModelRunner] = []

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

        model_runner = ModelRunner(
            model=mm,
            batch_size=batch_size,
            optimizer_kwargs=optimizer_kwargs,
        )
        mm_jobs.append(model_runner)

        for _ in range(3):
            # Introduce three different sets of mistakes in the MM (by removing 10% of connections)
            mm_with_mistakes = mm.with_missing_connections(fraction=0.1)

            model_runner = ModelRunner(
                model=mm_with_mistakes,
                batch_size=batch_size,
                optimizer_kwargs=optimizer_kwargs,
            )
            mm_jobs.append(model_runner)

            ml_model = best_mlp_factory(single_ko_dataset, device)
            hybrid_model = HybridModel(
                deepcopy(mm_with_mistakes), deepcopy(ml_model), device=device
            )

            model_runner = ModelRunner(
                model=hybrid_model,
                batch_size=batch_size,
                optimizer_kwargs=optimizer_kwargs,
            )

            hybrid_model_jobs.append(model_runner)

    # Run MM with mistakes
    with Pool(nodes=nr_processes) as pool:
        pool.map(
            partial(
                run_training_job,
                dataset_dict=dataset_dict,
                train_data_key_in_dict=train_dataset_name,
                val_data_key_in_dict=val_dataset_name,
                test_data_key_in_dict=test_dataset_name,
                simulation_parent_id=sim_run_id,
                nr_epochs=0,
                do_mlflow_model_logging=False,
            ),
            mm_jobs,
        )

    # Run hybrid models
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
            hybrid_model_jobs,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
