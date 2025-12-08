from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
from enum import Enum


class TidyDfColNames(Enum):
    """Enforce column names used in TidyDfResults."""

    MODEL = "model"
    GENE = "gene"
    TIME = "time"
    CONDITION = "condition"
    IS_IN_MM = "is_in_mm"
    EXPRESSION = "expression"
    GROUND_TRUTH = "ground_truth"
    MSE = "MSE"


class TidyDfResults:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _get_df_with_ground_truth_in_column(self) -> pd.DataFrame:
        """Return a dataframe where there is a column present called "ground_truth" which contains the ground truth expression values."""
        gt = self.df[self.df[TidyDfColNames.MODEL] == TidyDfColNames.GROUND_TRUTH][
            [
                TidyDfColNames.GENE,
                TidyDfColNames.TIME,
                TidyDfColNames.CONDITION,
                TidyDfColNames.EXPRESSION,
            ]
        ].rename(columns={TidyDfColNames.EXPRESSION: TidyDfColNames.GROUND_TRUTH})

        merged = self.df[
            self.df[TidyDfColNames.MODEL] != TidyDfColNames.GROUND_TRUTH
        ].merge(
            gt,
            on=[
                TidyDfColNames.GENE,
                TidyDfColNames.TIME,
                TidyDfColNames.CONDITION,
            ],
            how="left",
        )
        return merged

    def plot_mse_for_kos(self) -> sns.FacetGrid:
        """Make a plot that shows the MSE for each model. Split by if the genes are in the MM or not."""
        merged = self._get_df_with_gene_is_in_mm_column()

        mse_split = (
            merged.groupby([TidyDfColNames.MODEL, TidyDfColNames.IS_IN_MM])
            .apply(
                lambda g: mean_squared_error(
                    g[TidyDfColNames.GROUND_TRUTH], g[TidyDfColNames.EXPRESSION]
                ),
                include_groups=False,
            )
            .reset_index(name=TidyDfColNames.MSE)
        )

        g = sns.catplot(
            kind="bar",
            data=mse_split,
            col=TidyDfColNames.IS_IN_MM,
            y=TidyDfColNames.MSE,
            x=TidyDfColNames.MODEL,
            hue=TidyDfColNames.MODEL,
            aspect=0.5,
            height=5,
        )
        g.set_xticklabels(rotation=45, ha="right")
        return g

    def _get_df_with_gene_is_in_mm_column(self) -> pd.DataFrame:
        """Return DF that contains a column 'is_in_mm' that indicates if the gene is in the MM or not."""
        merged = self._get_df_with_ground_truth_in_column()
        total_nr_models = merged[TidyDfColNames.MODEL].nunique()
        merged[TidyDfColNames.IS_IN_MM] = merged.groupby(TidyDfColNames.GENE)[
            TidyDfColNames.MODEL
        ].transform(lambda x: x.nunique() == total_nr_models)
        return merged

    def plot_scatterplot(self):
        """Plot scatterplot of ground truth vs predicted expression. Split by if the gene is in the MM or not."""
        mse_split = self._get_df_with_gene_is_in_mm_column()
        return sns.relplot(
            data=mse_split,
            x=TidyDfColNames.GROUND_TRUTH,
            y=TidyDfColNames.EXPRESSION,
            hue=TidyDfColNames.MODEL,
            col=TidyDfColNames.IS_IN_MM,
        )

    def plot_scatterplot_per_ko(self):
        merged = self._get_df_with_ground_truth_in_column()
        return sns.relplot(
            data=merged,
            x=TidyDfColNames.GROUND_TRUTH,
            y=TidyDfColNames.EXPRESSION,
            hue=TidyDfColNames.MODEL,
            col=TidyDfColNames.CONDITION,
            col_wrap=3,
        )

    def plot_model_plots(self):
        """Plot lineplots of expression over time for each model, split by conditions."""
        g = sns.relplot(
            data=self.df,
            col=TidyDfColNames.MODEL,
            row=TidyDfColNames.CONDITION,
            hue=TidyDfColNames.GENE,
            x=TidyDfColNames.TIME,
            y=TidyDfColNames.EXPRESSION,
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
