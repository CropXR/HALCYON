from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd


class TidyDfResults:
    def __init__(self, df):
        self.df = df

    def _get_df_with_ground_truth_in_column(self):
        gt = self.df[self.df["model"] == "ground_truth"][
            ["gene", "time", "condition", "expression"]
        ].rename(columns={"expression": "ground_truth"})

        merged = self.df[self.df["model"] != "ground_truth"].merge(
            gt, on=["gene", "time", "condition"], how="left"
        )
        return merged

    def plot_mse_for_kos(self):
        merged = self._get_df_with_ground_truth_in_column()
        total_nr_models = merged["model"].nunique()
        merged["is_in_mm"] = merged.groupby("gene")["model"].transform(
            lambda x: x.nunique() == total_nr_models
        )

        mse_split = (
            merged.groupby(["model", "is_in_mm"])
            .apply(
                lambda g: mean_squared_error(g["ground_truth"], g["expression"]),
                include_groups=False,
            )
            .reset_index(name="MSE")
        )

        g = sns.catplot(
            kind="bar", data=mse_split, col="is_in_mm", y="MSE", x="model", hue="model"
        )
        g.set_xticklabels(rotation=90)
        return g

    def plot_scatterplot(self):
        merged = self._get_df_with_ground_truth_in_column()
        return sns.relplot(data=merged, x="ground_truth", y="expression", hue="model")

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
        g = sns.relplot(
            data=self.df,
            col="model",
            row="condition",
            hue="gene",
            x="time",
            y="expression",
            kind="line",
        )
        return

    def to_csv(self, filepath):
        self.df.to_csv(filepath, index=False)

    @classmethod
    def from_csv(cls, filepath):
        df = pd.read_csv(filepath)
        return cls(df)
