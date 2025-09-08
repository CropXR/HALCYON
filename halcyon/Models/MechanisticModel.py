import logging
import re
import libsbml
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

import torch
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model
from typing import List
from abc import abstractmethod
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model
from typing_extensions import Self

from torch import nn
import networkx as nx
from sympy import symbols, sympify


class MechanisticModel(Model):
    def __init__(self, modelled_gene_labels: List[str]):
        super().__init__(modelled_gene_labels)

    # @property
    # @abstractmethod
    # def regulatory_network(self) -> RegulatoryNetwork:
    #     """Return the regulatory network of the model."""
    #     pass


class PyTorchODE(MechanisticModel):
    alternative_max_name = "max_rate"

    def __init__(self, modelled_gene_labels, synthesis_dict, decay_dict, param_dict):
        super().__init__(modelled_gene_labels)
        self.synthesis_dict = synthesis_dict
        self.decay_dict = decay_dict
        self.param_dict = param_dict

    @classmethod
    def from_sbml(cls, sbml_path) -> Self:
        # Read sbml path
        # Import SBML file
        reader = libsbml.SBMLReader()
        document = reader.readSBML(sbml_path)  # type: ignore
        model = document.getModel()

        param_dict = dict()
        decay_expression_dict = dict()
        synthesis_expression_dict = dict()
        base_node_names = []

        for i in range(model.getNumReactions()):
            reaction = model.getReaction(i)
            base_node_name, sympy_formula, one_reactions_parameters = (
                cls.parse_one_reaction(reaction)
            )
            if base_node_name not in base_node_names:
                base_node_names.append(base_node_name)

            assert set(param_dict.keys()).isdisjoint(
                one_reactions_parameters.keys()
            ), "Parameter duplicates"
            param_dict.update(one_reactions_parameters)
            if reaction.getId().endswith("_degradation"):
                decay_expression_dict[base_node_name] = sympy_formula
            elif reaction.getId().endswith("_synthesis"):
                synthesis_expression_dict[base_node_name] = sympy_formula
            else:
                raise NameError(
                    f"Reaction {reaction.getId()} does not end with _synthesis or _degradation."
                )

        return cls(base_node_names, synthesis_expression_dict, decay_expression_dict, param_dict)

    @classmethod
    def parse_one_reaction(cls, reaction: libsbml.Reaction):
        base_node_name = (
            reaction.getId().replace("_degradation", "").replace("_synthesis", "")
        )
        kinetic_law = reaction.getKineticLaw()
        params = kinetic_law.getListOfParameters()
        formula = kinetic_law.getFormula()
        param_dict = dict()
        for param in params:
            if param.getId().startswith("binds") or param.getId().startswith("num"):
                continue

            # Get original parameter name
            param_id = param.getId()
            # if param_id == "max":
            #     param_id = cls.alternative_max_name
            #     # Replace 'max' with alternative_max_name in formula
            #     formula = re.sub(r'\bmax\b', cls.alternative_max_name, formula)

            if not param_id.endswith(base_node_name):
                original_name = param_id
                param_id = f"{param_id}_{base_node_name}"
            # Store parameter in dictionary
            assert (
                param_id not in param_dict
            ), f"Parameter {param_id} already exists in param_dict. Check for naming conflicts."
            param_dict[param_id] = param.getValue()
            # TODO check if this works properly
            formula = re.sub(
                r"\b" + re.escape(original_name) + r"\b", param_id, formula
            )

        sympy_formula = sp.parse_expr(formula)


        # ss_dict = {
        #         "Node_0": 0.1780818,
        #         "Node_1": 0.9222381,
        #         "Node_2": 0.1129278,
        #         "Node_3": 1.0,
        #         "Node_4": 0.1310036,
        #         "Node_5": 0.8210416,
        #         "cell": 1,
        #     }
        # sympy_formula.subs(param_dict | ss_dict)

        return base_node_name, sympy_formula, param_dict

    def forward(self, t: torch.Tensor, state: torch.Tensor):
        for index, name in zip(self.modelled_gene_labels, self.gene_index):
            ...

    def set_conditions(self, conditions: torch.Tensor) -> None:
        raise NotImplementedError("PyTorchODE does not implement set_conditions yet.")


class ComplexODEModel(MechanisticModel):
    alternative_max_name = "max_rate"

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

    @classmethod
    def from_sbml(cls, sbml_path: str) -> Self:
        reader = libsbml.SBMLReader()
        document = reader.readSBML(sbml_path)  # type: ignore
        model = document.getModel()
        print()
        # prepare parameter and function dictionaries
        param_dict = dict()
        ODE_dict = dict()

        # loop through reactions and parse them
        for i in range(model.getNumReactions()):
            reaction = model.getReaction(i)
            reaction_id = reaction.getId()  # .rstrip("_")

            expression, params = cls.__parse_one_reaction(reaction)
            # print(f"Parsed reaction {reaction_id}: {expression} with parameters {params}")
            ODE_dict[reaction_id] = expression

            # get separate parameters for each reaction, to be flattened later
            param_dict[reaction_id] = params

        # Flatten the parameter dictionary to match the expected format
        flat_param_dict = {}
        for node_name, params in param_dict.items():
            for pname, pval in params.items():
                flat_param_dict[pname] = pval

        # Get all node names from ODE_dict keys
        base_nodes = [
            node.replace("_degradation", "").replace("_synthesis", "")
            for node in ODE_dict.keys()
        ]  # get base node names without degradation/synthesis suffixes
        base_nodes = list(dict.fromkeys(base_nodes))  # remove duplicates

        # Combine synthesis and degradation equations for each node
        combined_ODE_dict = {}
        for node in base_nodes:
            synth_key = f"{node}_synthesis"
            degr_key = f"{node}_degradation"
            synth_expr = ODE_dict.get(synth_key, 0)
            degr_expr = ODE_dict.get(degr_key, 0)
            combined_ODE_dict[node] = synth_expr - degr_expr

        return cls(combined_ODE_dict, flat_param_dict)

    @classmethod
    def __parse_one_reaction(cls, reaction: libsbml.Reaction):
        """_summary_

        Args:
            reaction (libsbml.Reaction): _description_

        Returns:
            Tuple of [Sympy expression, all_parameters]
        """
        kinetic_law = reaction.getKineticLaw()
        params = kinetic_law.getListOfParameters()
        param_dict = {}
        base_node_name = (
            reaction.getId().replace("_degradation", "").replace("_synthesis", "")
        )
        formula = kinetic_law.getFormula()

        # print(f"Processing reaction {reaction.getId()} with formula: {formula}")

        # Process parameters and update formula such that parameter names have the node name as suffix
        for param in params:
            # Skip unwanted parameters
            if param.getId().startswith("binds") or param.getId().startswith("num"):
                continue

            # Get original parameter name
            orig_param_id = param.getId()

            # Handle special case for 'max'
            if orig_param_id == "max":
                orig_param_id = alternative_max_name
                # Replace 'max' with alternative_max_name in formula
                formula = re.sub(r"\bmax\b", alternative_max_name, formula)

            # Create the final parameter ID with node name
            final_param_id = orig_param_id
            if not final_param_id.endswith(base_node_name):
                final_param_id = f"{orig_param_id}_{base_node_name}"

            # Store parameter in dictionary
            param_dict[final_param_id] = param.getValue()

            # Update formula to use the final parameter ID
            # We need to be careful to replace whole words only
            formula = re.sub(
                r"\b" + re.escape(orig_param_id) + r"\b", final_param_id, formula
            )

        # Parse the updated formula
        expression = parse_expr(formula)

        return expression, param_dict

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
                input_val = torch.tensor(
                    input_val, dtype=state.dtype, device=state.device
                )
            state_dict[input_name] = input_val
        # Compute derivatives for each ODE variable
        for i, (var, ode_func) in enumerate(self.ODE_dict.items()):
            derivatives[i] = ode_func(state_dict)

        return derivatives

    def _build_network_from_self(self):
        dependencies = {}
        gene_names = list(self.ODE_dict.keys())
        input_names = (
            list(self.input_dict.keys())
            if self.input_dict is not None
            else ["light", "heat", "drought"]
        )
        for var, ODE in self.ODE_dict.items():
            symbolic_vars = symbols(gene_names + input_names)
            symbolic_expr = sympify(
                ODE(
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
        return RegulatoryNetwork(self._build_network_from_self())

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
