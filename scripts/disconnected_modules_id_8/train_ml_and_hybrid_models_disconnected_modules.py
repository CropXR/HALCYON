import multiprocess as mp
import random

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

    mlflow.set_experiment(experiment_id="8")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    sim_runs = mlflow.search_runs(
        experiment_ids=["8"],
        filter_string="tags.run_type = 'dataset'",
    )

    for sim_run_id in sim_runs["run_id"]:
        train_hybrid_and_ml_model_sim_run_id(sim_run_id)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
