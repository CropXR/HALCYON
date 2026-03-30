#!/usr/bin/env python
# coding: utf-8

import logging
import random

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

    # # Load from MLFlow
    print(mlflow.get_tracking_uri())
    [f"{experiment.name=}" for experiment in mlflow.search_experiments()]
    mlflow.set_experiment(experiment_id="4")
    mlflow.set_experiment_tag("Tool name", "HALCYON")

    # 20, 50, and 100 genes simulations
    simulation_run_ids = [
        "d660c1d337c34d98a349605d9b4d5b47",
        "19ba6fc0e8094a66a68f2f67cdca2162",
        "e56b1774f68349209f3b7e672158fa01",
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
    params_that_should_be_float = ["nr_genes", "params.lambda_l1"]
    for param_to_float in params_that_should_be_float:
        metrics_df[param_to_float] = metrics_df[param_to_float].astype("float")

    test_mse_col_name = "metrics.double_ko_final_mse_loss"
    train_mse_col_name = "metrics.train_single_ko_final_mse_loss"

    metrics_df["params.model_class_short"] = metrics_df[
        "params.model_class"
    ].str.replace("PhoenixAddAndProdLayers", "Phoenix")

    param_cols = [
        "params.model_class_short",
        "params.nr_layers",
        "params.hidden_size",
        "params.synthesis_only_positive",
        "params.lambda_l1",
    ]
    col_labels = ["Model", "# Layers", "Hidden size", "Reg+", "λ"]

    metrics_df["long_model_name"] = (
        metrics_df["params.model_class"]
        + "_nr_layers"
        + metrics_df["params.nr_layers"].fillna("")
        + "_hidden_size"
        + metrics_df["params.hidden_size"].fillna("")
        + "_synth_only_pos"
        + metrics_df["params.synthesis_only_positive"].fillna("").astype("str")
        + "_λ"
        + metrics_df["params.lambda_l1"].astype("str")
    )
    order = (
        metrics_df.groupby("long_model_name")[test_mse_col_name]
        .mean()
        .sort_values()
        .index
    )
    n = len(order)

    param_table = (
        metrics_df.drop_duplicates("long_model_name")
        .set_index("long_model_name")
        .loc[order, param_cols]
    )

    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

    ax_table = fig.add_subplot(gs[0])
    ax_plot = fig.add_subplot(gs[1])

    ax_table.set_xlim(0, len(col_labels))
    ax_table.set_ylim(-0.5, n - 0.5)
    ax_table.invert_yaxis()
    ax_table.axis("off")

    # column headers
    for col_idx, col_label in enumerate(col_labels):
        ax_table.text(
            col_idx + 0.5,
            -0.6,
            col_label,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
        )

    # alternating row backgrounds + values
    for row_idx, model in enumerate(order):
        if row_idx % 2 == 0:
            ax_table.axhspan(row_idx - 0.5, row_idx + 0.5, color="lightgrey", alpha=0.3)
            ax_plot.axhspan(row_idx - 0.5, row_idx + 0.5, color="lightgrey", alpha=0.3)

        for col_idx, col in enumerate(param_cols):
            val = param_table.loc[model, col]
            ax_table.text(
                col_idx + 0.5, row_idx, str(val), ha="center", va="center", fontsize=7
            )

    # strip plot
    sns.stripplot(
        data=metrics_df,
        x=test_mse_col_name,
        y="long_model_name",
        hue="nr_genes",
        order=order,
        jitter=False,
        ax=ax_plot,
    )

    ax_plot.set_ylabel("")
    ax_plot.set_xlim(left=0)
    ax_plot.set_xlabel("Double knockout MSE loss")
    ax_plot.xaxis.grid(True, alpha=0.3)
    ax_plot.set_yticklabels([])
    ax_plot.set_yticks([])

    plt.savefig(
        "comparing_models_from_hyperparam_search_across_datasets.svg",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    main()
