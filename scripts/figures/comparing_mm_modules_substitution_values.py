import logging
import random

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from dotenv import load_dotenv


def main():
    seed = 174
    # for CPU
    torch.manual_seed(seed)
    # for all GPUs
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,  # Important in Jupyter to reset config
    )

    print(mlflow.get_tracking_uri())
    [f"{experiment.name=}" for experiment in mlflow.search_experiments()]
    mlflow.set_experiment("Halcyon comparing models and datasets")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    # 20, 50, and 100 genes simulations
    simulation_run_ids = [
        "912f193730964b508691ee4ec81d66be",
        "dcbfe325396842f3ba4ec365436a0fe4",
        "aa7b3f101bb94054a85dde642cf28ad2",
    ]
    run_df_list = []
    for id in simulation_run_ids:
        parent_run_id_search_string = f"tags.mlflow.parentRunId= '{id}'"
        run_df_list.append(
            mlflow.search_runs(
                filter_string=parent_run_id_search_string, output_format="pandas"
            )
        )

    metrics_df = pd.concat(run_df_list)
    parent_run_params = dict()
    for parent_run_id in metrics_df["tags.mlflow.parentRunId"].unique():
        run = mlflow.get_run(parent_run_id)
        parent_run_params[parent_run_id] = run.data.params
    params_df = pd.DataFrame.from_dict(parent_run_params, orient="index")
    params_df.index.name = "tags.mlflow.parentRunId"
    metrics_df = metrics_df.merge(params_df, on="tags.mlflow.parentRunId", how="left")
    metrics_df.head()

    params_that_should_be_float = [
        "nr_genes",
        "params.fill_value_for_gene_not_in_model",
    ]
    for param_to_float in params_that_should_be_float:
        metrics_df[param_to_float] = metrics_df[param_to_float].astype("float")

    test_mse_col_name = "metrics.double_ko_final_mse_loss"
    train_mse_col_name = "metrics.train_single_ko_final_mse_loss"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sns.swarmplot(
        data=metrics_df,
        y=train_mse_col_name,
        x="nr_genes",
        hue="params.fill_value_for_gene_not_in_model",
        dodge=True,
        ax=axes[0],
    )
    axes[0].set_title("Single Knockout")
    axes[0].set_ylim(0, 0.16)
    axes[0].set_ylabel("MSE")
    axes[0].get_legend().remove()

    sns.swarmplot(
        data=metrics_df,
        y=test_mse_col_name,
        x="nr_genes",
        hue="params.fill_value_for_gene_not_in_model",
        dodge=True,
        ax=axes[1],
    )
    axes[1].set_title("Double Knockout")
    axes[1].set_ylim(0, 0.16)
    axes[1].legend(
        bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0, title="fill_value"
    )

    plt.tight_layout()
    plt.savefig("mm_substitution_values_comparison.svg", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
