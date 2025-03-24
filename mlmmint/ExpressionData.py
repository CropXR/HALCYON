from typing import List
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import torch
import seaborn as sns


class ExpressionData:
    """Store RNA seq expression data (from simulator)"""

    time_name = "Time"
    gene_name = "Gene"
    expression_name = "Expression"

    def __init__(self, expression_data: pd.DataFrame):
        """Rows are genes and columns are samples"""
        self.df = expression_data

    def match_genes_to_list(self, genes: list):
        """Keep only the specified genes in the data"""
        return ExpressionData(self.df.loc[genes])

    @property
    def genes(self) -> List:
        return list(self.df.index)

    def _add_one_gene_expression(self, gene_name: str, expression: float):
        """Add an exra gene with a constant expression to the DF. Mainly for testing purposes"""
        new_df = self.df.copy()
        new_df.loc[gene_name] = expression
        return ExpressionData(new_df)

    def _get_initial_states(self) -> np.ndarray:
        """Return the initial states of the genes

        Is assumed to be in first column of the dataframe
        """
        return self.df.iloc[:, 0].values  # type: ignore

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

    def plot_expressions(self, axis=None) -> Axes:
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
        )
