import networkx as nx


class GeneNetwork:
    def __init__(self, network: nx.DiGraph):
        self.graph = network

    def get_ingoing_interactions(self, gene_name: str):
        """Get the edges that influence the given gene.

        Parameters:
        - gene_name (str): The name of the gene.

        """
        return list(self.graph.in_edges(gene_name, data=True))

    def get_all_interactions(self):
        """Get all edges in the network."""
        return list(self.graph.edges(data=True))

    @property
    def genes(self):
        """Note: these might differ from the order of ExpressionData.genes

        To fix, perhaps do look into merging these two classes into one.
        """

        return list(self.graph.nodes)

    def add_edge(self, edge: tuple, attributes: dict, contains_new_genes=False):
        """
        Add an edge to the network.

        Parameters:
        - edge (tuple): A tuple containing the source and target genes.
        - attributes (dict): A dictionary containing the attributes of the edge.
        - contains_new_genes (bool): If True, add any new genes to the network.

        """
        if not contains_new_genes:
            assert all(
                gene in self.genes for gene in edge
            ), "All genes in the edge must be present in the network if not_contains_new_genes=False."
        self.graph.add_nodes_from(edge)
        self.graph.add_edge(*edge, **attributes)
