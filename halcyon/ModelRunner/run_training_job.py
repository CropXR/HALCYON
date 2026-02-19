from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.ModelRunner.ModelRunner import ModelRunner

import logging


def run_training_job(
    job: ModelRunner,
    dataset_dict: dict[str, ExpressionDataWithConditions],
    train_data_key_in_dict: str,
    simulation_parent_id: str = None,
    do_mlflow_model_logging: bool = False,
):
    """Runs one training job. Can be called from mp.Pool to run multiple in parallel"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(message)s",
        force=True,
    )
    try:
        output = job.train_and_predict_for_mlflow(
            dataset_dict,
            train_data_key_in_dict,
            simulation_parent_id,
            do_mlflow_model_logging,
        )
    except AssertionError as e:
        logging.error(f"Error occurred while running job: {e}")
        output = {"error": str(e)}
    return output
