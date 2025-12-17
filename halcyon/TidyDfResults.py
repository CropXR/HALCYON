import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd
from typing import Callable
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


class TidyDfResults:
    # Store column names here
    MODEL_COL = "model"
    GENE_COL = "gene"
    TIME_COL = "time"
    CONDITION_COL = "condition"
    GENE_IS_IN_MM_COL = "gene_is_in_mm"
    EXPRESSION_COL = "expression"
    GROUND_TRUTH_COL = "ground_truth"
    MSE_COL = "MSE"
    R2_COL = "R2"
    KO_IN_MM_COL = "ko_in_mm"

    def __init__(self, df: pd.DataFrame):
        # Check if the columns are correct
        assert set(df.columns) == {
            self.GENE_COL,
            self.TIME_COL,
            self.EXPRESSION_COL,
            self.MODEL_COL,
            self.CONDITION_COL,
        }
        self.df = df

    def _get_df_with_ground_truth_in_column(self) -> pd.DataFrame:
        """Return a dataframe where there is a column present called "ground_truth" which contains the ground truth expression values."""
        gt = self.df[self.df[self.MODEL_COL] == self.GROUND_TRUTH_COL][
            [
                self.GENE_COL,
                self.TIME_COL,
                self.CONDITION_COL,
                self.EXPRESSION_COL,
            ]
        ].rename(columns={self.EXPRESSION_COL: self.GROUND_TRUTH_COL})

        merged = self.df[self.df[self.MODEL_COL] != self.GROUND_TRUTH_COL].merge(
            gt,
            on=[
                self.GENE_COL,
                self.TIME_COL,
                self.CONDITION_COL,
            ],
            how="left",
        )
        return merged

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
        merged = self._get_df_with_ground_truth_in_column()

        gene_is_in_mm = self._flag_groups_with_all_models(
            merged, col_to_group=self.GENE_COL
        )
        merged[self.GENE_IS_IN_MM_COL] = gene_is_in_mm

        return merged

    def _flag_groups_with_all_models(self, merged: pd.DataFrame, col_to_group: str):
        """Return a boolean Series indicating if all models are present for each group defined by col_to_group."""
        total_nr_models = merged[self.MODEL_COL].nunique()
        gene_is_in_mm = merged.groupby(col_to_group)[self.MODEL_COL].transform(
            lambda x: x.nunique() == total_nr_models
        )

        return gene_is_in_mm

    def plot_r2(self, split_by_condition_in_mm=True):
        """Make a plot that shows the R2 for each model. Split by if the genes are in the MM or not."""
        # Do not show per-gene r2 because these can be massively different
        groupby_list = [self.MODEL_COL, self.GENE_IS_IN_MM_COL]

        if split_by_condition_in_mm:
            groupby_list.append(self.CONDITION_COL)
        r2_split = self._groupby_and_apply_func(
            r2_score, self.R2_COL, groupby_list=groupby_list
        )
        if split_by_condition_in_mm:
            condition_in_mm = self._flag_groups_with_all_models(
                r2_split, col_to_group=self.CONDITION_COL
            )
            r2_split[self.KO_IN_MM_COL] = condition_in_mm

        g = sns.catplot(
            kind="bar",
            data=r2_split,
            col=self.GENE_IS_IN_MM_COL,
            col_order=[True, False],
            row=self.KO_IN_MM_COL if split_by_condition_in_mm else None,
            row_order=[True, False] if split_by_condition_in_mm else None,
            y=self.R2_COL,
            x=self.MODEL_COL,
            hue=self.MODEL_COL,
            aspect=1,
            height=5,
            sharey=True,
            margin_titles=True,
        )
        self._rotate_x_tick_labels(g)
        return g

    def plot_mse_per_gene(
        self, split_by_condition_in_mm=True, y_log_scale=True
    ) -> sns.FacetGrid:
        """Make a plot that shows the MSE for each model. Split by if the genes are in the MM or not."""
        groupby_list = [
            self.MODEL_COL,
            self.GENE_COL,
            self.GENE_IS_IN_MM_COL,
            self.CONDITION_COL,
        ]

        mse_split = self._groupby_and_apply_func(
            mean_squared_error, self.MSE_COL, groupby_list=groupby_list
        )

        if split_by_condition_in_mm:
            condition_in_mm = self._flag_groups_with_all_models(
                mse_split, col_to_group=self.CONDITION_COL
            )
            mse_split[self.KO_IN_MM_COL] = condition_in_mm

        g = sns.catplot(
            kind="boxen",
            data=mse_split,
            # col=self.GENE_IS_IN_MM_COL,
            # col_order=[True, False],
            col=self.KO_IN_MM_COL if split_by_condition_in_mm else None,
            col_order=[True, False] if split_by_condition_in_mm else None,
            y=self.MSE_COL,
            hue=self.MODEL_COL,
            x=self.GENE_IS_IN_MM_COL,
            aspect=1,
            height=5,
            # margin_titles=True if split_by_condition_in_mm else None,
            # sharey="row" if split_by_condition_in_mm else True,
            log_scale=y_log_scale,
        )
        self._rotate_x_tick_labels(g)
        return g

    def _rotate_x_tick_labels(self, g, rotation=45):
        for ax in g.axes.flatten():
            for label in ax.get_xticklabels():
                label.set_rotation(rotation)
                label.set_horizontalalignment("right")
        g.figure.tight_layout()

    def kde_plot(self, subsample_first=False):
        """Show KDE of points in ground truth vs predicted plot. Set subsample=True to speed up calculation of KDE by subsampling DF first."""
        df = self._get_df_with_gene_is_in_mm_column()
        condition_in_mm = self._flag_groups_with_all_models(
            df, col_to_group=self.CONDITION_COL
        )
        df[self.KO_IN_MM_COL] = condition_in_mm
        if subsample_first:
            df = df.groupby(
                [self.KO_IN_MM_COL, self.GENE_IS_IN_MM_COL, self.MODEL_COL]
            ).sample(n=10000, replace=True, random_state=42)
        return sns.displot(
            data=df,
            kind="kde",
            x=self.GROUND_TRUTH_COL,
            y=self.EXPRESSION_COL,
            col=self.GENE_IS_IN_MM_COL,
            row=self.KO_IN_MM_COL,
            hue=self.MODEL_COL,
            levels=3,
            thresh=0.01,
        )

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
        self.df.to_parquet(filepath, index=False)
