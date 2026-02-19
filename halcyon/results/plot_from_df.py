import textwrap

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from halcyon.results.DfColumnNames import DfColumnNames
from halcyon.results.manipulate_results_df import df_with_ground_truth_as_kind_of_model


def plot_model_plots(df_with_model_name: pd.DataFrame):
    """Plot lineplots of expression over time for each model, split by conditions."""
    assert (
        DfColumnNames.MODEL_COL in df_with_model_name.columns
    ), "Provide model name first"

    df_with_gt_as_model = df_with_ground_truth_as_kind_of_model(df_with_model_name)

    g = sns.relplot(
        data=df_with_gt_as_model,
        col=DfColumnNames.MODEL_COL,
        row=DfColumnNames.CONDITION_COL,
        hue=DfColumnNames.GENE_COL,
        x=DfColumnNames.TIME_COL,
        y=DfColumnNames.EXPRESSION_COL,
        kind="line",
        facet_kws={"margin_titles": True},
    )
    return g


def plot_per_gene_mse_heatmap(
    df_with_model_name: pd.DataFrame,
    cmap="viridis",
    robust_cmap=False,
    norm_function=LogNorm,
) -> sns.FacetGrid:
    """Show per gene MSE in heatmap for various models

    Args:
        df_with_model_name (pd.DataFrame): Dataframe to be plotted, should contain a column MODEL_COL that shows the model name.
        cmap (str, optional): Colourmap. Defaults to "viridis".
        robust (bool, optional): How to determine colour mapping. If false, span full min to max. If true, span 1st to 99th percentile. Defaults to False.

    """
    assert (
        DfColumnNames.MODEL_COL in df_with_model_name.columns
    ), "Provide model name first"
    groupby_list = [
        DfColumnNames.MODEL_COL,
        DfColumnNames.GENE_COL,
        DfColumnNames.CONDITION_COL,
    ]
    mse_df = (
        df_with_model_name.groupby(groupby_list)
        .apply(
            lambda g: mean_squared_error(
                g[DfColumnNames.GROUND_TRUTH_COL], g[DfColumnNames.EXPRESSION_COL]
            ),
            include_groups=False,
        )
        .reset_index(name=DfColumnNames.MSE_COL)
    )

    if robust_cmap:
        vmin, vmax = np.percentile(mse_df[DfColumnNames.MSE_COL].dropna(), [1, 99])
        norm = norm_function(vmin=vmin, vmax=vmax)
    else:
        norm = norm_function(
            vmin=mse_df[DfColumnNames.MSE_COL].min(),
            vmax=mse_df[DfColumnNames.MSE_COL].max(),
        )

    # Use correlations between conditions to order columns
    full_pivot = mse_df.pivot(
        index=[DfColumnNames.GENE_COL, DfColumnNames.MODEL_COL],
        columns=DfColumnNames.CONDITION_COL,
        values=DfColumnNames.MSE_COL,
    )
    corr = full_pivot.corr()
    linkage = sch.linkage(corr, method="average")
    col_order = full_pivot.columns[sch.leaves_list(linkage)]

    # Use how often gene shows up (i.e. if it is in MM or not) to order rows
    gene_order = mse_df[DfColumnNames.GENE_COL].value_counts().index

    # Do the actual plotting
    g = sns.FacetGrid(
        mse_df, col=DfColumnNames.MODEL_COL, margin_titles=True, aspect=0.4, height=10
    )

    def _plot_heatmap_func(data, **kwargs):
        df_pivot = data.pivot(
            index=DfColumnNames.GENE_COL,
            columns=DfColumnNames.CONDITION_COL,
            values=DfColumnNames.MSE_COL,
        ).reindex(index=gene_order, columns=col_order)
        sns.heatmap(data=df_pivot, cmap=cmap, norm=norm, cbar=False, **kwargs)

    g.map_dataframe(_plot_heatmap_func)

    # Remove all gene and condition names
    for ax in g.axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # add one shared colourbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    g.figure.colorbar(mappable, ax=g.axes, fraction=0.02, pad=0.01, label="MSE")

    return g


def rotate_x_tick_labels(g: sns.FacetGrid, rotation=45):
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
            label.set_horizontalalignment("right")
    g.figure.tight_layout()


def wrap_titles(g: sns.FacetGrid, width=40):
    """Wrap titles to be max length width"""
    for ax in g.axes.flat:
        title = ax.get_title()
        wrapped = "\n".join(textwrap.wrap(title, width=width))
        ax.set_title(wrapped)
