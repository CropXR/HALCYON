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
from halcyon.Models import PhoenixAddAndProdLayers, PyTorchODE, HybridModel

from halcyon.Models.BaseLineModels import AlwaysDerivativeZero
from halcyon.Models.Losses import MSELossPlusMachineLearningL2


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


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
    single_ko_dataset, double_ko_dataset, time_scaling_factor = (
        sim_output.create_datasets()
    )
    dataset_dict = {
        "single_ko": single_ko_dataset,
        "double_ko": double_ko_dataset,
    }

    device = cycle(devices)
    # # Define hyperparams
    # device = cycle(['cuda:0', 'cuda:1'])
    # batch_size = 32
    # nr_processes = 4
    # nr_epochs = 50
    # hidden_size_list = [10, 20, 100]
    # do_only_pos_list = [True]
    optimizer_kwargs = {"lr": optimizer_lr}

    # Create train jobs
    train_jobs: list[ModelRunner] = []

    # TODO how to identify which genes to select for the mechanistic model?
    node_that_should_be_in_model = "Node_18"
    for i, module in enumerate(sim_output.network._clustering):
        if node_that_should_be_in_model in module:
            print("Module:", i)
            print("Size:", len(module))
            genes_to_select = list(sim_output.network._clustering[i])
            break

    mm = PyTorchODE.from_sbml(
        sim_output.data_path / "my_network.xml",
        condition_names=[],
        from_gnw=True,
        device=next(device),
    )
    # Select only genes from specific module and set timescaler
    mm.select_genes(
        genes_to_select,
        goldstandard_signed_file_for_expression_substitution=sim_output.data_path
        / f"{sim_output.network_name}_goldstandard_signed.tsv",
    )
    mm.set_timescaler(time_scaling_factor)
    # Creating MM trainer even though it cannot be trained
    mm_trainer = ModelRunner(
        mm,
        nr_epochs=0,
        batch_size=batch_size,
    )

    train_jobs.append(mm_trainer)

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

            # Create hybrid model with MM and with add and prod
            hybrid_model = HybridModel(
                deepcopy(mm), deepcopy(ml_model), device=next(device)
            )
            for model in [ml_model, hybrid_model]:
                # my_loss = MSELossPlusMachineLearningL2(lambda_param=0.5)
                model_runner = ModelRunner(
                    model=model,
                    batch_size=batch_size,
                    nr_epochs=nr_epochs,
                    optimizer_kwargs=optimizer_kwargs,
                )
                train_jobs.append(model_runner)

    # Always predict zero model
    baseline_model = AlwaysDerivativeZero(
        modelled_gene_labels=single_ko_dataset.genes,
        condition_names=single_ko_dataset.all_environments,
    )

    baseline_trainer = ModelRunner(
        baseline_model,
        batch_size=batch_size,
        nr_epochs=0,
    )
    train_jobs.append(baseline_trainer)

    for job in train_jobs:
        logging.info(f"{job=}")

    # Do the training
    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=nr_processes) as pool:
        pool.map(
            partial(
                run_training_job,
                dataset_dict=dataset_dict,
                train_data_key_in_dict="single_ko",
                simulation_parent_id=sim_run_id,
                do_mlflow_model_logging=False,
            ),
            train_jobs,
        )


if __name__ == "__main__":
    typer.run(main)
