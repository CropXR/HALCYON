from typing import List
import networkx as nx


class RegulatoryNetwork:
    def __init__(self, network: nx.DiGraph):
        self._graph = network

    def get_ingoing_interactions(self, gene_name: str):
        """Get interactions that influence a certain gene

        Args:
            gene_name (str): Name of target gene

        Returns:
            in_edges (list): List of all edges that are ingoing to gene (with edge attributes included)
        """
        return list(self._graph.in_edges(gene_name, data=True))

    def get_all_interactions(self, with_data: bool = True):
        """Return all interaction (edges) in the network.

        Args:
            with_data (bool, optional): If set to true (default) include the edge attributes as dict. Defaults to True.

        Returns:
            edges (list): List of all edges in graph
        """
        return list(self._graph.edges(data=with_data))

    @property
    def node_names(self):
        """Note: these might differ from the order of ExpressionData.genes."""
        return list(self._graph.nodes)

    def add_edge(self, edge: tuple, attributes: dict, contains_new_genes=False):
        """Add an edge to the network.

        Args:
            edge (tuple): A tuple containing the source and target genes.
            attributes (dict): A dictionary containing the attributes of the edge.
            contains_new_genes (bool, optional): If True, add any new genes to the network. Defaults to False.
        """
        if not contains_new_genes:
            assert all(
                gene in self.node_names for gene in edge
            ), "All genes in the edge must be present in the network if not_contains_new_genes=False."
        self._graph.add_nodes_from(edge)
        self._graph.add_edge(*edge, **attributes)

    def select_subset(self, nodes: List[str]):
        """Return only subset of the network.

        Args:
            nodes (List[str]): A list of nodes to include in the subset.

        Returns:
            subset_network (RegulatoryNetwork): Network that contains only the specified nodes
        """
        return RegulatoryNetwork(self._graph.subgraph(nodes).copy())
