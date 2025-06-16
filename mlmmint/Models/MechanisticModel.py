import torch
from mlmmint.RegulatoryNetwork import RegulatoryNetwork
from mlmmint.Models.Model import Model
from typing import List
from torch import nn
import networkx as nx
from sympy import symbols, sympify


class MechanisticModel(Model):
    def __init__(self, modelled_gene_labels: List[str]):
        super().__init__(modelled_gene_labels)

class ComplexODEMechanisticModel(MechanisticModel):
    def __init__(self, ODE_dict: dict, input_dict: dict, param_dict: dict):
        """
        Initialize a complex mechanistic model.
        param regulatory_network The regulatory network to use.
        param parameter_dict A dictionary of parameters to use in the model.
        """
        # regulatory_network = self.build_network_from_ODE_dict(ODE_DICT)
        super().__init__(list(ODE_dict.keys()))
        
        self.ODE_dict = ODE_dict
        self.input_dict = input_dict
        self.param_dict = param_dict
        # self.regulatory_network = regulatory_network

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

        # Prepare state dict with gene and input values as tensors
        state_dict = dict(zip(self.ODE_dict.keys(), torch.unbind(state)))
        # Update state_dict with input values at time t (ensure tensor output)
        for input_name, input_func in self.input_dict.items():
            input_val = input_func(t)
            # If input_val is not a tensor, convert it
            if not isinstance(input_val, torch.Tensor):
                input_val = torch.tensor(input_val, dtype=state.dtype, device=state.device)
            state_dict[input_name] = input_val
        # Compute derivatives for each ODE variable
        for i, (var, ode_func) in enumerate(self.ODE_dict.items()):
            derivatives[i] = ode_func(state_dict)
        
        


        # # Add inputs to the state
        # state = torch.cat((state, torch.tensor([self.regulatory_network.nodes[node_name]["time_series_func"](t) for node_name in self.regulatory_network.nodes if self.regulatory_network.nodes[node_name]["type"] == "input"])))

        # # convert the state to a dictionary for easier access
        # state_dict = {gene: state[i] for i, gene in enumerate(self.modelled_gene_labels)}

        # # print("state_dict",state_dict)

        # # Iterate over all nodes in the regulatory network
        # # for i, node in enumerate(self.regulatory_network.nodes()):
        # #     # Calculate the derivative using the Nodes' ODE function and the current state
        # #     derivatives[node] = self.regulatory_network.nodes[node]["ODE"](state_dict)
        # for i, node in enumerate(self.regulatory_network.nodes):
        #     # print(i,node)
        #     # Check if the node is a Network node
        #     if self.regulatory_network.nodes[node]["type"] == "network":
        #         derivatives[i] = self.regulatory_network.nodes[node]["ODE"](state_dict) #calculate the ODE output for each ODE node
        return derivatives
    
    def build_network_from_ODE_dict(self):
        dependencies = {}
        gene_names = list(self.ODE_dict.keys())
        input_names = list(self.input_dict.keys()) if self.input_dict is not None else ["light", "heat", "drought"]
        for var, ODE in self.ODE_dict.items():
            symbolic_vars = symbols(gene_names + input_names)
            symbolic_expr = sympify(
                ODE({v: symbolic_vars[i] for i, v in enumerate(gene_names)} |
                    {name: symbolic_vars[len(gene_names) + j] for j, name in enumerate(input_names)})
            )
            dependencies[var] = [
                str(sym) for sym in symbolic_expr.free_symbols
                if str(sym) in gene_names or str(sym) in input_names
            ]

        net = nx.DiGraph()
        for var, ODE in self.ODE_dict.items():
            net.add_node(var, ODE=ODE, type="network")

        for input_node, time_series_func in self.input_dict.items():
            net.add_node(input_node, time_series_func=time_series_func, type="input")
        
        for var, deps in dependencies.items():
            for d in deps:
                net.add_edge(d, var)
        return net
    
    @property
    def regulatory_network(self):
        """Return the regulatory network of the model."""
        return RegulatoryNetwork(self.build_network_from_ODE_dict())


        
    


class TorchLinearODE(MechanisticModel):
    def __init__(self, regulatory_network: RegulatoryNetwork):
        super().__init__(regulatory_network)

        # Initialize the adjacency/weight matrix
        self.weight_matrix = torch.zeros(
            (self.n_genes, self.n_genes), dtype=torch.float32
        )

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
