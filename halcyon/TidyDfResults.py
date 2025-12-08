from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd


class TidyDfResults:
    def __init__(self, df):
        self.df = df

    def _get_df_with_ground_truth_in_column(self) -> pd.DataFrame:
        """Return a dataframe where there is a column present called "ground_truth" which contains the ground truth expression values."""
        gt = self.df[self.df["model"] == "ground_truth"][
            ["gene", "time", "condition", "expression"]
        ].rename(columns={"expression": "ground_truth"})

        merged = self.df[self.df["model"] != "ground_truth"].merge(
            gt, on=["gene", "time", "condition"], how="left"
        )
        return merged

    def plot_mse_for_kos(self) -> sns.FacetGrid:
        """Make a plot that shows the MSE for each model. Split by if the genes are in the MM or not."""
        merged = self._get_df_with_gene_is_in_mm_column()

        mse_split = (
            merged.groupby(["model", "is_in_mm"])
            .apply(
                lambda g: mean_squared_error(g["ground_truth"], g["expression"]),
                include_groups=False,
            )
            .reset_index(name="MSE")
        )

        g = sns.catplot(
            kind="bar",
            data=mse_split,
            col="is_in_mm",
            y="MSE",
            x="model",
            hue="model",
            aspect=0.5,
            height=5,
        )
        g.set_xticklabels(rotation=45, ha="right")
        return g

    def _get_df_with_gene_is_in_mm_column(self) -> pd.DataFrame:
        """Return DF that contains a column 'is_in_mm' that indicates if the gene is in the MM or not."""
        merged = self._get_df_with_ground_truth_in_column()
        total_nr_models = merged["model"].nunique()
        merged["is_in_mm"] = merged.groupby("gene")["model"].transform(
            lambda x: x.nunique() == total_nr_models
        )
        return merged

    def plot_scatterplot(self):
        """Plot scatterplot of ground truth vs predicted expression. Split by if the gene is in the MM or not."""
        mse_split = self._get_df_with_gene_is_in_mm_column()
        return sns.relplot(
            data=mse_split,
            x="ground_truth",
            y="expression",
            hue="model",
            col="is_in_mm",
        )

    def plot_scatterplot_per_ko(self):
        merged = self._get_df_with_ground_truth_in_column()
        return sns.relplot(
            data=merged,
            x="ground_truth",
            y="expression",
            hue="model",
            col="condition",
            col_wrap=3,
        )

    def plot_model_plots(self):
        """Plot lineplots of expression over time for each model, split by conditions."""
        g = sns.relplot(
            data=self.df,
            col="model",
            row="condition",
            hue="gene",
            x="time",
            y="expression",
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
