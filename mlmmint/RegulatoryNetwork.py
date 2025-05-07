from typing import List
import networkx as nx


class RegulatoryNetwork:
    def __init__(self, network: nx.DiGraph):
        self._graph = network

    def get_ingoing_interactions(self, gene_name: str):
        """Get the edges that influence the given gene.

        Parameters:
        - gene_name (str): The name of the gene.

        """
        return list(self._graph.in_edges(gene_name, data=True))

    def get_all_interactions(self):
        """Get all edges in the network."""
        return list(self._graph.edges(data=True))

    @property
    def node_names(self):
        """Note: these might differ from the order of ExpressionData.genes."""
        return list(self._graph.nodes)
    
    @property
    def nodes(self):
        """Get all nodes in the network."""
        return self._graph.nodes

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
                gene in self.node_names for gene in edge
            ), "All genes in the edge must be present in the network if not_contains_new_genes=False."
        self._graph.add_nodes_from(edge)
        self._graph.add_edge(*edge, **attributes)

    def select_subset(self, nodes: List[str]):
        """Return only subset of the network.

        Parameters:
        - nodes (list): A list of nodes to include in the subset.

        """
        return RegulatoryNetwork(self._graph.subgraph(nodes).copy())
