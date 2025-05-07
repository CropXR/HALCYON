import torch
from mlmmint.RegulatoryNetwork import RegulatoryNetwork
from mlmmint.Models.Model import Model

from torch import nn


class MechanisticModel(Model):
    def __init__(self, network: RegulatoryNetwork):
        super().__init__(network.nodes)

class ComplexMechanisticModel(MechanisticModel):
    def __init__(self, regulatory_network: RegulatoryNetwork, parameter_dict: dict):
        """
        Initialize a complex mechanistic model.
        param regulatory_network The regulatory network to use.
        param parameter_dict A dictionary of parameters to use in the model.
        """
        super().__init__(regulatory_network)
        self.parameter_dict = parameter_dict
        self.regulatory_network = regulatory_network

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Get the derivative at a given time point.
        Parameters:
            t (torch.Tensor): The time point at which to evaluate the ODE.
            state (torch.Tensor): The current state of the system.
        Returns:
            torch.Tensor: The derivative of the state at the given time point.
        """
        # Initialize the derivatives
        derivatives = torch.zeros_like(state)


        # Add inputs to the state
        state = torch.cat((state, torch.tensor([self.regulatory_network.nodes[node_name]["time_series_func"](t) for node_name in self.regulatory_network.nodes if self.regulatory_network.nodes[node_name]["type"] == "input"])))

        # convert the state to a dictionary for easier access
        state_dict = {gene: state[i] for i, gene in enumerate(self.modelled_gene_labels)}

        # print("state_dict",state_dict)

        # Iterate over all nodes in the regulatory network
        # for i, node in enumerate(self.regulatory_network.nodes()):
        #     # Calculate the derivative using the Nodes' ODE function and the current state
        #     derivatives[node] = self.regulatory_network.nodes[node]["ODE"](state_dict)
        for i, node in enumerate(self.regulatory_network.nodes):
            # print(i,node)
            # Check if the node is a Network node
            if self.regulatory_network.nodes[node]["type"] == "network":
                derivatives[i] = self.regulatory_network.nodes[node]["ODE"](state_dict) #calculate the ODE output for each ODE node
        return derivatives

        
    


class TorchLinearODE(MechanisticModel):
    def __init__(self, regulatory_network: RegulatoryNetwork):
        super().__init__(regulatory_network)

        # Initialize the adjacency/weight matrix
        self.weight_matrix = torch.zeros(
            (self.n_genes, self.n_genes), dtype=torch.float32
        )
        self.gene_index = {gene: i for i, gene in enumerate(self.modelled_gene_labels)}

        for (
            from_gene,
            to_gene,
            property_dict,
        ) in regulatory_network.get_all_interactions():
            if from_gene in self.gene_index and to_gene in self.gene_index:
                from_idx = self.gene_index[from_gene]
                to_idx = self.gene_index[to_gene]
                self.weight_matrix[to_idx, from_idx] = property_dict["influence"]
        self.weight_matrix = nn.Parameter(self.weight_matrix, requires_grad=True)

    def forward(self, t, state):
        return self.weight_matrix @ state

    def _make_weight_matrix_zeroes(self):
        """Initialize the weight matrix by setting everything to 0."""
        self.weight_matrix = nn.Parameter(
            torch.zeros_like(self.weight_matrix, dtype=torch.float32),
            requires_grad=True,
        )
