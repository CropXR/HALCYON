from typing import List
import numpy as np
import pandas as pd
import torch


class ExpressionData:
    """Store RNA seq expression data (from simulator)"""

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
        return torch.Tensor(self.df.values)
