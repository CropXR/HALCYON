import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.axes import Axes


from typing import List


class ExpressionData:
    """Store RNA seq expression data (from a single simulation run)"""

    time_name = "Time"
    gene_name = "Gene"
    expression_name = "Expression"

    def __init__(self, expression_data: pd.DataFrame):
        """Rows are genes and columns are samples"""
        self.df = expression_data

    def match_genes_to_list(self, genes: list):
        """Keep only the specified genes in the data"""
        return ExpressionData(self.df.loc[genes])

    def subsample_time_points(self, fraction: float):
        """Keep only fraction of time points. The time points are evenly spaced.

        Used for testing how well models work when there are fewer time points. Modifies the object in place.
        """
        step = max(1, round(1 / fraction))
        self.df = self.df.iloc[:, ::step]

    @property
    def genes(self) -> List:
        return list(self.df.index)

    def scale_time_series(self):
        self.df.columns = self.df.columns / self.df.columns.max()

    def _get_time_points(self) -> np.ndarray:
        """Return the time points of the samples"""
        return self.df.columns.values.astype(float)

    def as_tensor(self) -> torch.Tensor:
        return torch.Tensor(
            self.df.values,
        )

    def _return_long_df(self) -> pd.DataFrame:
        return self.df.melt(
            ignore_index=False, value_name=self.expression_name, var_name=self.time_name
        ).reset_index(names=self.gene_name)

    def plot_expressions(self, axis=None, legend: bool = True, **kwargs) -> Axes:
        """Plot gene expression levels.

        This method does not call plt.show() to allow integration into larger
        subplot layouts. Call plt.show() separately to display the figure.
        """
        df_to_plot = self._return_long_df()
        return sns.lineplot(
            data=df_to_plot,
            x=self.time_name,
            y=self.expression_name,
            hue=self.gene_name,
            ax=axis,
            legend=legend,
            **kwargs,
        )
