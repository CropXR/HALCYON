import logging
import torch
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model
from typing import List
from abc import abstractmethod
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model

from torch import nn
import networkx as nx
from sympy import symbols, sympify


class MechanisticModel(Model):
    def __init__(self, modelled_gene_labels: List[str]):
        super().__init__(modelled_gene_labels)


class ComplexODEModel(MechanisticModel):
    def __init__(self, ode_dict: dict, input_dict: dict, param_dict: dict):
        """
        Initialize a complex mechanistic model.
        param regulatory_network The regulatory network to use.
        param parameter_dict A dictionary of parameters to use in the model.
        """
        super().__init__(modelled_gene_labels=list(ode_dict.keys()))

        self.ode_dict = ode_dict
        self.input_dict = input_dict
        self.param_dict = {
            k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
            for k, v in param_dict.items()
        }

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
        state_dict = dict(zip(self.ode_dict.keys(), torch.unbind(state)))
        # Update state_dict with input values at time t (ensure tensor output)
        for input_name, input_func in self.input_dict.items():
            input_val = input_func(t)
            # If input_val is not a tensor, convert it
            if not isinstance(input_val, torch.Tensor):
                input_val = torch.tensor(
                    input_val, dtype=state.dtype, device=state.device
                )
            state_dict[input_name] = input_val
        # Compute derivatives for each ODE variable
        for i, (var, ode_func) in enumerate(self.ode_dict.items()):
            derivatives[i] = ode_func(state_dict)

        return derivatives

    def get_regulatory_network(self) -> RegulatoryNetwork:
        """Return the regulatory network of the model."""
        dependencies = {}
        gene_names = list(self.ode_dict.keys())
        input_names = list(self.input_dict.keys())
        for var, ode in self.ode_dict.items():
            symbolic_vars = symbols(gene_names + input_names)
            symbolic_expr = sympify(
                ode(
                    {v: symbolic_vars[i] for i, v in enumerate(gene_names)}
                    | {
                        name: symbolic_vars[len(gene_names) + j]
                        for j, name in enumerate(input_names)
                    }
                )
            )
            dependencies[var] = [
                str(sym)
                for sym in symbolic_expr.free_symbols
                if str(sym) in gene_names or str(sym) in input_names
            ]

        net = nx.DiGraph()
        for var, ode in self.ode_dict.items():
            # TODO is it needed to store these things in the network?
            net.add_node(var, ode=ode, type="network")

        for input_node, time_series_func in self.input_dict.items():
            net.add_node(input_node, time_series_func=time_series_func, type="input")

        for var, deps in dependencies.items():
            for d in deps:
                net.add_edge(d, var)

        return RegulatoryNetwork(net)

    def set_conditions(self, conditions):
        logging.warning("Currently ComplexODEModel does not parse the conditions")
        pass


class TorchLinearODE(MechanisticModel):
    def __init__(self, regulatory_network: RegulatoryNetwork):
        super().__init__(list(regulatory_network.nodes))
        self.regulatory_network = regulatory_network

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

    def forward(self, t, state) -> torch.Tensor:
        return torch.bmm(
            self.weight_matrix.unsqueeze(0),
            state.unsqueeze(2),
        ).squeeze(2)

    def _make_weight_matrix_zeroes(self):
        """Initialize the weight matrix by setting everything to 0."""
        self.weight_matrix = nn.Parameter(
            torch.zeros_like(self.weight_matrix, dtype=torch.float32),
            requires_grad=True,
        )

    def set_conditions(self, conditions):
        logging.warning("Currently TorchLinearODE does not parse the conditions")
        pass
