from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.ModelRunner.ModelRunner import ModelRunner

import logging


def run_training_job(
    job: ModelRunner,
    dataset_dict: dict[str, ExpressionDataWithConditions],
    train_data_key_in_dict: str,
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
            do_mlflow_model_logging=do_mlflow_model_logging,
        )
    except AssertionError as e:
        logging.error(f"Error occurred while running job: {e}")
        output = {"error": str(e)}
    return output
