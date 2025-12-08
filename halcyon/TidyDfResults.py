from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd


class TidyDfResults:
    # Store column names here
    MODEL_COL = "model"
    GENE_COL = "gene"
    TIME_COL = "time"
    CONDITION_COL = "condition"
    IS_IN_MM_COL = "is_in_mm"
    EXPRESSION_COL = "expression"
    GROUND_TRUTH_COL = "ground_truth"
    MSE_COL = "MSE"

    def __init__(self, df: pd.DataFrame):
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

    def plot_mse_for_kos(self) -> sns.FacetGrid:
        """Make a plot that shows the MSE for each model. Split by if the genes are in the MM or not."""
        merged = self._get_df_with_gene_is_in_mm_column()

        mse_split = (
            merged.groupby([self.MODEL_COL, self.IS_IN_MM_COL])
            .apply(
                lambda g: mean_squared_error(
                    g[self.GROUND_TRUTH_COL], g[self.EXPRESSION_COL]
                ),
                include_groups=False,
            )
            .reset_index(name=self.MSE_COL)
        )

        g = sns.catplot(
            kind="bar",
            data=mse_split,
            col=self.IS_IN_MM_COL,
            y=self.MSE_COL,
            x=self.MODEL_COL,
            hue=self.MODEL_COL,
            aspect=0.5,
            height=5,
        )
        g.set_xticklabels(rotation=45, ha="right")
        return g

    def _get_df_with_gene_is_in_mm_column(self) -> pd.DataFrame:
        """Return DF that contains a column 'is_in_mm' that indicates if the gene is in the MM or not."""
        merged = self._get_df_with_ground_truth_in_column()
        total_nr_models = merged[self.MODEL_COL].nunique()
        merged[self.IS_IN_MM_COL] = merged.groupby(self.GENE_COL)[
            self.MODEL_COL
        ].transform(lambda x: x.nunique() == total_nr_models)
        return merged

    def plot_scatterplot(self):
        """Plot scatterplot of ground truth vs predicted expression. Split by if the gene is in the MM or not."""
        mse_split = self._get_df_with_gene_is_in_mm_column()
        return sns.relplot(
            data=mse_split,
            x=self.GROUND_TRUTH_COL,
            y=self.EXPRESSION_COL,
            hue=self.MODEL_COL,
            col=self.IS_IN_MM_COL,
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
