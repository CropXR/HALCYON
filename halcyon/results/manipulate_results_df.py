import logging
from pathlib import Path
import mlflow

import pandas as pd
from typing import List, Tuple


from halcyon.results.DfColumnNames import DfColumnNames


def create_tidy_df_from_mlflow_runs(runs: List[mlflow.entities.Run]):
    """Create tidy df from a list of mlflow runs"""
    dfs = []
    for run in runs:
        run_id = run.info.run_id
        logging.info(f"Downloading artifacts for run {run_id}")
        parent_run_id = run.data.tags["mlflow.parentRunId"]
        artifacts = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="evaluation"
        )
        for artifact in Path(artifacts).iterdir():
            if artifact.is_file() and artifact.suffix == ".parquet":
                df = pd.read_parquet(artifact)
                df[DfColumnNames.RUN_ID_COL] = run_id
                df[DfColumnNames.PARENT_RUN_ID_COL] = parent_run_id
                dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    assert set(full_df.columns) == {
        DfColumnNames.RUN_ID_COL,
        DfColumnNames.GENE_COL,
        DfColumnNames.TIME_COL,
        DfColumnNames.EXPRESSION_COL,
        DfColumnNames.GROUND_TRUTH_COL,
        DfColumnNames.CONDITION_COL,
        DfColumnNames.PARENT_RUN_ID_COL,
    }
    return full_df


def split_df_into_single_and_double_ko(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split this dataframe into one with single knockouts and double knockouts.

    Can be used if you want to plot these predictions separately

    Returns:
        Tuple[df, df]: First element is df with single KOs, second element is df with double KOs
    """
    nr_of_ko_genes = df[DfColumnNames.CONDITION_COL].str.count("Node_")
    single_ko = df[nr_of_ko_genes == 1]
    double_ko = df[nr_of_ko_genes == 2]
    return single_ko, double_ko


def add_model_name_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """For a dataframe which only contains mlflow run ids, return one where these are converted to strings representing the underlying model."""
    run_id_to_model_dict = {}
    for run_id in df[DfColumnNames.RUN_ID_COL].unique():
        model_name = run_id_to_model_name(run_id)
        run_id_to_model_dict[run_id] = model_name
    df_with_model_name = df.assign(
        **{
            DfColumnNames.MODEL_COL: df[DfColumnNames.RUN_ID_COL].map(
                run_id_to_model_dict
            )
        }
    )
    return df_with_model_name


def run_id_to_model_name(run_id: str) -> str:
    """Convert mlflow run_id to a more human readable model name by looking up the parameters of the run and merging them into one big string."""
    logging.info(f"Getting model name for run_id {run_id}")
    run = mlflow.get_run(run_id)
    out_string_list = []
    for key, value in run.data.params.items():
        out_string_list.append(f"{key}_{value}")
    out_string = "__".join(out_string_list)
    return out_string


def df_with_ground_truth_as_kind_of_model(in_df: pd.DataFrame) -> pd.DataFrame:
    """For some plotting, pretend that the ground truth expression was a form of model.

    This function creates a new dataframe where the ground truth is added as a
    'ground truth' model predictions. This way, it can be plotted together with the predictions.
    """
    df_with_ground_truth_as_model = in_df.copy()

    df_with_ground_truth_as_model[DfColumnNames.EXPRESSION_COL] = (
        df_with_ground_truth_as_model[DfColumnNames.GROUND_TRUTH_COL]
    )
    df_with_ground_truth_as_model[DfColumnNames.MODEL_COL] = (
        DfColumnNames.GROUND_TRUTH_COL
    )
    df_expanded = pd.concat([in_df, df_with_ground_truth_as_model], ignore_index=True)
    return df_expanded
