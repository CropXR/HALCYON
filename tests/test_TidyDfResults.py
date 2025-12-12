import pandas as pd
import pytest
from halcyon.TidyDfResults import TidyDfResults
import numpy as np

from sklearn.metrics import mean_squared_error


@pytest.fixture
def tidy_df():
    # Settings
    genes = [f"Node_{i}" for i in range(4)]
    time_points = [0, 1, 2]
    models = ["ground_truth", "model_A", "model_B"]
    mechanistic_genes = ["Node_0", "Node_2"]
    conditions = ["KO: Node_0", "KO: Node_2"]

    # Generate ground truth
    gt_data = []
    for cond in conditions:
        for gene in genes:
            for t in time_points:
                gt_data.append([gene, t, np.random.rand(), "ground_truth", cond])
    df_gt = pd.DataFrame(
        gt_data,
        columns=[
            TidyDfResults.GENE_COL,
            TidyDfResults.TIME_COL,
            TidyDfResults.EXPRESSION_COL,
            TidyDfResults.MODEL_COL,
            TidyDfResults.CONDITION_COL,
        ],
    )

    # Other models with random deviations
    other_data = []
    for model in ["model_A", "model_B"]:
        for cond in conditions:
            for gene in genes:
                for t in time_points:
                    gt_expr = df_gt[
                        (df_gt[TidyDfResults.GENE_COL] == gene)
                        & (df_gt[TidyDfResults.TIME_COL] == t)
                        & (df_gt[TidyDfResults.CONDITION_COL] == cond)
                    ][TidyDfResults.EXPRESSION_COL].values[0]
                    expr = gt_expr + np.random.normal(scale=0.1)
                    other_data.append([gene, t, expr, model, cond])
    df_other = pd.DataFrame(
        other_data,
        columns=[
            TidyDfResults.GENE_COL,
            TidyDfResults.TIME_COL,
            TidyDfResults.EXPRESSION_COL,
            TidyDfResults.MODEL_COL,
            TidyDfResults.CONDITION_COL,
        ],
    )

    # Mechanistic model (perfect match for Node_0 and Node_2)
    mech_data = []
    for model in ["Mech_model"]:
        for cond in conditions:
            for gene in mechanistic_genes:
                for t in time_points:
                    gt_expr = df_gt[
                        (df_gt.gene == gene)
                        & (df_gt.time == t)
                        & (df_gt.condition == cond)
                    ][TidyDfResults.EXPRESSION_COL].values[0]
                    mech_data.append([gene, t, gt_expr, model, cond])
    df_mechanistic = pd.DataFrame(
        mech_data,
        columns=[
            TidyDfResults.GENE_COL,
            TidyDfResults.TIME_COL,
            TidyDfResults.EXPRESSION_COL,
            TidyDfResults.MODEL_COL,
            TidyDfResults.CONDITION_COL,
        ],
    )

    # Combine all
    df_full = pd.concat([df_gt, df_other, df_mechanistic], ignore_index=True)
    return TidyDfResults(df_full)


def test_calculate_mse(tidy_df):
    mse_df = tidy_df._groupby_and_apply_func(
        mean_squared_error,
        TidyDfResults.MSE_COL,
        groupby_list=[
            TidyDfResults.MODEL_COL,
            TidyDfResults.GENE_COL,
            TidyDfResults.GENE_IS_IN_MM_COL,
            TidyDfResults.CONDITION_COL,
        ],
    )
    # For mech model it should have MSE=0.
    assert (
        mse_df[
            (mse_df[TidyDfResults.MODEL_COL] == "Mech_model")
            & (mse_df[TidyDfResults.GENE_IS_IN_MM_COL] == True)
        ][TidyDfResults.MSE_COL].values[0]
        == 0.0
    )
    # For the others it should be > 0
    assert all(
        mse_df[(mse_df[TidyDfResults.MODEL_COL] != "Mech_model")][
            TidyDfResults.MSE_COL
        ].values
        > 0
    )
