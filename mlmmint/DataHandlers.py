from mlmmint.ExpressionData import ExpressionData
from mlmmint.GeneNetwork import GeneNetwork


class GeneExpressionDataHandler:
    """Handles all data preprocessing, e.g. ensuring the genes in expressiondata match the genes or doing gene normalisation"""

    def __init__(self, expression_data: ExpressionData, network: GeneNetwork):
        self.data = expression_data
        self.network = network

    def filter_matching_genes(self):
        """When performed the genes in the expression data will be trimmed to only include genes present in the network"""
        self.data = self.data.keep_genes_in_list(self.network.genes)

    def get_genes(self):
        return self.data.genes
