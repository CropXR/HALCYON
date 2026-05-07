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

    mlflow.set_experiment(experiment_id="9")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    # 50 genes medium noise
    sim_run_id = "f02bf732b2ba4817b254bd5174f1549b"
    for fraction in [0.25, 0.5]:
        train_hybrid_and_ml_model_sim_run_id(
            sim_run_id, time_points_training_fraction=fraction
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
