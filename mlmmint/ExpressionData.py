import pandas as pd


class ExpressionData:
    """Store RNA seq expression data (from simulator)"""

    def __init__(self, expression_data: pd.DataFrame):
        """Rows are genes and columns are samples"""
        self.df = expression_data

    def keep_genes_in_list(self, genes: list):
        """Keep only the specified genes in the data"""
        return ExpressionData(self.df[self.df.index.isin(genes)])

    @property
    def genes(self):
        return list(self.df.index)

    def _add_one_gene_expression(self, gene_name: str, expression: float):
        """Add an exra gene with a constant expression to the DF. Mainly for testing purposes"""
        new_df = self.df.copy()
        new_df.loc[gene_name] = expression
        return ExpressionData(new_df)
