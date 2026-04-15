import multiprocess as mp
import logging
import random
import os

import mlflow
import torch
import numpy as np
from dotenv import load_dotenv

from halcyon.ModelRunner import train_hybrid_and_ml_model_sim_run_id


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
    sim_runs = mlflow.search_runs(
        experiment_ids=["7"],
        filter_string="tags.run_type = 'dataset' and params.noise_level = 'MEDIUM'",
    )

    # Start with smallest network first
    sim_runs["params.nr_genes"] = sim_runs["params.nr_genes"].astype(int)
    sim_runs = sim_runs.sort_values("params.nr_genes", ascending=True)

    for sim_run_id in sim_runs["run_id"]:
        train_hybrid_and_ml_model_sim_run_id(sim_run_id, nr_processes=16)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
