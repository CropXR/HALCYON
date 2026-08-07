import re
from typing import List
from typing_extensions import Self
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import leidenalg
import igraph as ig


class RegulatoryNetwork:
    def __init__(self, network: nx.DiGraph):
        self._graph: nx.DiGraph = network
        self._clustering: list[set] = []

    def __eq__(self, other: Self):
        return nx.is_isomorphic(self._graph, other._graph)

    def do_clustering(self):
        ig_graph = ig.Graph.from_networkx(self._graph)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            seed=174,
        )
        # convert back to node names
        self._clustering = [
            [ig_graph.vs[i]["_nx_name"] for i in community] for community in partition
        ]

        # self._clustering = algorithms.leiden(self._graph, seed=12).communities

    def get_ingoing_interactions(self, gene_name: str) -> List:
        """Get interactions that influence a certain gene

        Args:
            gene_name (str): Name of target gene

        Returns:
            in_edges (list): List of all edges that are ingoing to gene (with edge attributes included)
        """
        return list(self._graph.in_edges(gene_name, data=True))

    def get_all_interactions(self, with_data: bool = True) -> List:
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

    @property
    def nodes(self):
        """Get all nodes in the network."""
        return self._graph.nodes

    def get_disconnected_modules(self) -> list[set]:
        """For disconnected modules, return list of sets of genes that are in the same module."""
        return list(nx.weakly_connected_components(self._graph))

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

    def select_subset(self, nodes: List[str]) -> Self:
        """Return only subset of the network.

        Args:
            nodes (List[str]): A list of nodes to include in the subset.

        Returns:
            subset_network (RegulatoryNetwork): Network that contains only the specified nodes
        """
        return RegulatoryNetwork(self._graph.subgraph(nodes).copy())

    def _get_node_colours(self):
        if not self._clustering:
            return None
        else:
            palette = sns.color_palette("tab20", len(self._clustering))
            node_to_colour = {}
            for i, community in enumerate(self._clustering):
                for gene in community:
                    node_to_colour[gene] = palette[i]
            return [node_to_colour[n] for n in self._graph.nodes()]

    def get_ko_to_target_distance(self, gene_name: str, ko_string: str) -> float:
        """For a gene name and a double KO. Calculate the sum of distance from the KO gene to the target gene.

        Mainly used to see which gene - double KO combos were hardest to predict.
        """
        kos = re.findall(r"Node_\d+", ko_string)
        assert (
            len(kos) == 2
        ), f"Only working / used for double KOs currently. But was privded with KO string: {ko_string}\n This gives {kos=}"

        depths = []
        for ko in kos:
            try:
                d = nx.shortest_path_length(self._graph, ko, gene_name)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = None
            depths.append(d)
        if any(d is None for d in depths):
            return None
        else:
            return sum(depths)

    def plot(
        self,
        ax=None,
        layout_func=nx.kamada_kawai_layout,
        separate_components=False,
        component_spacing=2.0,
        with_labels=True,
        node_size=300,
        arrowsize=10,
        width=1,
    ):
        """Plot the network using networkx's draw function.

        Args:
            ax (optional): Matplotlib ax to plot on. Defaults to None.
            layout_func (_type_, optional): networkx layout function to use for positioning nodes. Defaults to nx.kamada_kawai_layout.
            separate_components (bool, optional): Whether to plot disconnected components in the plot next to each other. Defaults to False.
            component_spacing (float, optional): Spacing between components if plotting separate. Defaults to 2.0.
        """
        if separate_components:
            pos = {}
            x_offset = 0
            for component in nx.weakly_connected_components(self._graph):
                subgraph = self._graph.subgraph(component)
                sub_pos = layout_func(subgraph)
                for node, (x, y) in sub_pos.items():
                    pos[node] = (x + x_offset, y)
                x_offset += component_spacing
        else:
            pos = layout_func(self._graph)

        # Extract edge colors from the attribute
        edge_colors = [self._graph[u][v]["color"] for u, v in self._graph.edges()]
        node_colours = self._get_node_colours()

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        nx.draw(
            self._graph,
            pos,
            ax=ax,
            connectionstyle="arc3,rad=0.2",
            with_labels=with_labels,
            arrows=True,
            edge_color=edge_colors,
            node_color=node_colours,
            node_size=node_size,
            arrowsize=arrowsize,
            width=width,
        )
        return fig, ax

    @classmethod
    def from_stderr_gnw(
        cls,
        stderr: str,
        n_genes: int,
        network_name: str,
        keep_multiplicative=False,
    ) -> Self:
        """Create RegulatoryNetwork from the stderror text output that is generated by GNW

        Args:
            stderr (str): raw stderr text output from GNW
            n_genes (int): number of genes in network (used to check if lines match this number)
            network_name (str): Name of network, used to identify relevant lines in stderr
            keep_multiplicative (bool, optional): Whether to keep multiplicative interactions. Defaults to False.
        """
        lines = stderr.splitlines()
        network_lines = [i for i in lines if i.startswith(network_name)]
        assert len(network_lines) == n_genes

        G = nx.DiGraph()
        for line in network_lines:
            _, target_str, regulator_str = line.split(": ")
            if regulator_str == "no inputs":
                continue
            for regulator in regulator_str.split(" + "):
                if regulator.startswith("~"):
                    # Inhibitory interaction
                    regulator = regulator[1:]
                    colour = "red"
                else:
                    colour = "green"
                regulator = regulator.replace(")", "").replace("(", "")
                if keep_multiplicative:
                    G.add_edge(regulator, target_str, color=colour)
                elements_of_complex = re.split(r"[~*]", regulator)
                for element in elements_of_complex:
                    if keep_multiplicative:
                        G.add_edge(element, regulator, color="black")
                    else:
                        G.add_edge(element, target_str, color=colour)
        return cls(G)
