from pathlib import Path
from itertools import cycle
import logging
from copy import deepcopy
from functools import partial

import mlflow

from pathos.multiprocessing import ProcessingPool as Pool

from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.ModelRunner.ModelRunner import ModelRunner
from halcyon.Simulators import SimulationResult
from halcyon.Models import best_mlp_factory, HybridModel, PyTorchODE


def run_mm_sim_run_id(
    sim_run_id: str,
    nr_processes: int = 8,
    nr_epochs: int = 0,
    batch_size: int = 32,
    lr: float = 0.01,
    devices: cycle = cycle(["cuda:1", "cuda:0"]),
):
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
        if "Node_30" not in genes_to_select:
            continue
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


def train_hybrid_and_ml_model_sim_run_id(
    sim_run_id: str,
    nr_processes: int = 8,
    nr_epochs: int = 2000,
    batch_size: int = 32,
    lr: float = 0.01,
    devices: cycle = cycle(["cuda:1", "cuda:0"]),
):
    """Perform ML and hybrid model training based on a simulation run id

    Runs a single MLP and runs all hybrid models for every possible MM subset in the simulation network

    Args:
        sim_run_id (str): Simulation run id to load data from.
        nr_processes (int, optional): Number of parallel processes to use for training. Defaults to 8.
        nr_epochs (int, optional): Number of epochs to train for. Defaults to 2000.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        devices (_type_, optional): Devices to use for training. Defaults to cycle(["cuda:1", "cuda:0"]).
    """
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
    device = next(devices)
    ml_model = best_mlp_factory(single_ko_dataset, device)

    model_runner = ModelRunner(
        model=ml_model,
        batch_size=batch_size,
        optimizer_kwargs=optimizer_kwargs,
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

        ml_model = best_mlp_factory(single_ko_dataset, device)
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


def prepare_dataset(sim_run_id):
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

    return (
        sim_output,
        single_ko_dataset,
        time_scaling_factor,
        train_dataset_name,
        val_dataset_name,
        test_dataset_name,
        dataset_dict,
    )


def run_training_job(
    job: ModelRunner,
    dataset_dict: dict[str, ExpressionDataWithConditions],
    train_data_key_in_dict: str,
    nr_epochs: int,
    val_data_key_in_dict: str | None = None,
    test_data_key_in_dict: str | None = None,
    simulation_parent_id: str = None,
    do_mlflow_model_logging: bool = False,
):
    """Runs one training job. Can be called from mp.Pool to run multiple in parallel"""
    try:
        output = job.train_and_predict_for_mlflow(
            dataset_dict=dataset_dict,
            simulation_parent_id=simulation_parent_id,
            train_data_key_in_dict=train_data_key_in_dict,
            val_data_key_in_dict=val_data_key_in_dict,
            test_data_key_in_dict=test_data_key_in_dict,
            nr_epochs=nr_epochs,
            do_mlflow_model_logging=do_mlflow_model_logging,
        )
    except Exception as e:
        logging.error(f"Error occurred while running job: {e}")
        output = {"error": str(e)}
    return output
