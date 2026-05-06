import multiprocess as mp
import logging
import random
import os
import pandas as pd

import mlflow
import torch
import numpy as np
from dotenv import load_dotenv

from halcyon.ModelRunner.run_training_job import run_mm_sim_run_id


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

    df_list = []

    for noise_level in ["HIGH", "MEDIUM", "LOW"]:
        sim_runs = mlflow.search_runs(
            experiment_ids=["7"],
            filter_string=f"tags.run_type = 'dataset' and params.noise_level = '{noise_level}' and params.nr_genes = '50'",
        )
        df_list.append(sim_runs)

    sim_runs = pd.concat(df_list)

    for sim_run_id in sim_runs["run_id"]:
        run_mm_sim_run_id(sim_run_id)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
