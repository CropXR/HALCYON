import logging
from pathlib import Path
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd
from typing import Callable, List, Tuple

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from typing_extensions import Self

from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions


class TidyDfResults:
    """Class to easily make plots from tidy dataframe (and sometimes convert to other formats).
    Dataframe keeps track of predictions of different models for different genes under different conditions at different times.
    """

    # Store column names here
    MODEL_COL = "model"
    GENE_COL = "gene"
    TIME_COL = "time"
    CONDITION_COL = "condition"
    GENE_IS_IN_MM_COL = "gene_is_in_mm"
    EXPRESSION_COL = "predicted_expression"
    GROUND_TRUTH_COL = "ground_truth"
    MSE_COL = "MSE"
    R2_COL = "R2"
    RUN_ID_COL = "run_id"

    def __init__(self, df: pd.DataFrame):
        # Check if the columns are correct
        assert set(df.columns) == {
            self.RUN_ID_COL,
            self.GENE_COL,
            self.TIME_COL,
            self.EXPRESSION_COL,
            self.GROUND_TRUTH_COL,
            self.CONDITION_COL,
        }
        self.df = df

    @classmethod
    def from_mlflow_runs(cls, runs: List[mlflow.entities.Run]) -> Self:
        """Create TidyDfResults from a list of mlflow runs"""
        dfs = []
        for run in runs:
            run_id = run.info.run_id
            logging.info(f"Downloading artifacts for run {run_id}")
            artifacts = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="evaluation"
            )
            for artifact in Path(artifacts).iterdir():
                if artifact.is_file() and artifact.suffix == ".parquet":
                    df = pd.read_parquet(artifact)
                    df[cls.RUN_ID_COL] = run_id
                    dfs.append(df)

        return cls(pd.concat(dfs, ignore_index=True))

    def _groupby_and_apply_func(
        self, func_to_apply: Callable, col_title: str, groupby_list: list[str]
    ):
        merged = self._get_df_with_gene_is_in_mm_column()
        df_split_and_function_applied = (
            merged.groupby(groupby_list)
            .apply(
                lambda g: func_to_apply(
                    g[self.GROUND_TRUTH_COL], g[self.EXPRESSION_COL]
                ),
                include_groups=False,
            )
            .reset_index(name=col_title)
        )

        return df_split_and_function_applied

    def _get_df_with_gene_is_in_mm_column(self) -> pd.DataFrame:
        """Return DF that contains a column 'is_in_mm' that indicates if the gene is in the MM or not."""

        gene_is_in_mm = self._flag_groups_with_all_models(
            self.df, col_to_group=self.GENE_COL
        )
        self.df[self.GENE_IS_IN_MM_COL] = gene_is_in_mm

        return merged

    def _flag_groups_with_all_models(self, merged: pd.DataFrame, col_to_group: str):
        """Return a boolean Series indicating if all models are present for each group defined by col_to_group."""
        total_nr_models = merged[self.MODEL_COL].nunique()
        gene_is_in_mm = merged.groupby(col_to_group)[self.MODEL_COL].transform(
            lambda x: x.nunique() == total_nr_models
        )

        return gene_is_in_mm

    def _rotate_x_tick_labels(self, g, rotation=45):
        for ax in g.axes.flatten():
            for label in ax.get_xticklabels():
                label.set_rotation(rotation)
                label.set_horizontalalignment("right")
        g.figure.tight_layout()

    def plot_scatterplot(self):
        """Plot scatterplot of ground truth vs predicted expression. Split by if the gene is in the MM or not."""
        df = self._get_df_with_gene_is_in_mm_column()
        condition_in_mm = self._flag_groups_with_all_models(
            df, col_to_group=self.CONDITION_COL
        )
        df[self.KO_IN_MM_COL] = condition_in_mm

        return sns.relplot(
            data=df,
            x=self.GROUND_TRUTH_COL,
            y=self.EXPRESSION_COL,
            col=self.GENE_IS_IN_MM_COL,
            row=self.KO_IN_MM_COL,
            hue=self.MODEL_COL,
            marker=".",
        )

    def plot_scatterplot_per_ko(self):
        """Plot scatterplot of ground truth vs predicted expression. Split by condition (KO)."""
        merged = self._get_df_with_ground_truth_in_column()
        return sns.relplot(
            data=merged,
            x=self.GROUND_TRUTH_COL,
            y=self.EXPRESSION_COL,
            hue=self.MODEL_COL,
            col=self.CONDITION_COL,
            col_wrap=3,
        )

    def per_gene_mse_heatmap(self, cmap="viridis", robust=False) -> sns.FacetGrid:
        groupby_list = [
            self.MODEL_COL,
            self.GENE_COL,
            self.CONDITION_COL,
        ]
        mse_df = self._groupby_and_apply_func(
            mean_squared_error, self.MSE_COL, groupby_list=groupby_list
        )
        if robust:
            vmin, vmax = np.percentile(mse_df[self.MSE_COL].dropna(), [1, 99])
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = LogNorm(
                vmin=mse_df[self.MSE_COL].min(),
                vmax=mse_df[self.MSE_COL].max(),
            )

        # Use correlations between conditions to order columns
        full_pivot = mse_df.pivot(
            index=[self.GENE_COL, self.MODEL_COL],
            columns=self.CONDITION_COL,
            values=self.MSE_COL,
        )
        corr = full_pivot.corr()
        linkage = sch.linkage(corr, method="average")
        col_order = full_pivot.columns[sch.leaves_list(linkage)]

        # Use how often gene shows up (i.e. if it is in MM or not) to order rows
        gene_order = mse_df[self.GENE_COL].value_counts().index

        # Do the actual plotting
        g = sns.FacetGrid(
            mse_df, col=self.MODEL_COL, margin_titles=True, aspect=0.4, height=10
        )

        def _plot_heatmap_func(data, **kwargs):
            df_pivot = data.pivot(
                index=self.GENE_COL, columns=self.CONDITION_COL, values=self.MSE_COL
            ).reindex(index=gene_order, columns=col_order)
            sns.heatmap(data=df_pivot, cmap=cmap, norm=norm, cbar=False, **kwargs)

        g.map_dataframe(_plot_heatmap_func)

        # Remove all gene and condition names
        for ax in g.axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        # add ONE shared colourbar
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        g.figure.colorbar(mappable, ax=g.axes, fraction=0.02, pad=0.01, label="MSE")

        return g

    def plot_model_plots(self):
        """Plot lineplots of expression over time for each model, split by conditions."""
        g = sns.relplot(
            data=self.df,
            col=self.MODEL_COL,
            row=self.CONDITION_COL,
            hue=self.GENE_COL,
            x=self.TIME_COL,
            y=self.EXPRESSION_COL,
            kind="line",
        )
        return g

    def to_csv(self, filepath):
        """Save to csv file."""
        self.df.to_csv(filepath, index=False)

    @classmethod
    def from_csv(cls, filepath):
        df = pd.read_csv(filepath)
        return cls(df)

    @classmethod
    def from_parquet(cls, filepath):
        df = pd.read_parquet(filepath)
        return cls(df)

    def to_parquet(self, filepath):
        """Save to parquet file."""
        self.df.to_parquet(filepath, index=False)
