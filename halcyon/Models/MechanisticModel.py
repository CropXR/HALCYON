import logging
import libsbml
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

import torch
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model

from torch import nn


class MechanisticModel(Model):
    def __init__(self, network: RegulatoryNetwork):
        super().__init__(network.node_names)


class SBMLImportClassDummy(MechanisticModel):
    alternative_max_name = "max_rate"

    def __init__(self, regulatory_network: RegulatoryNetwork):
        super().__init__(regulatory_network)

    @classmethod
    def from_sbml(cls, sbml_path: str):
        reader = libsbml.SBMLReader()
        document = reader.readSBML(sbml_path)  # type: ignore
        model = document.getModel()

        param_dict = dict()
        reaction_dict = dict()

        for i in range(model.getNumReactions()):
            reaction = model.getReaction(i)
            reaction_id = reaction.getId()
            expression, params = cls.__parse_one_reaction(reaction)
            reaction_dict[reaction_id] = expression
            # Currently just get separate parameters for each reaction
            param_dict[reaction_id] = params

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
        param_out_dict = dict()
        for param in params:
            if param.getId().startswith("binds") or param.getId().startswith("num"):
                # Skip parameters that are not relevant for the model
                continue
            elif param.getId() == "max":
                # Rename max to max_rate to avoid conflicts with built-in max function
                param.setId(cls.alternative_max_name)
            param_out_dict[param.getId()] = param.getValue()
        formula = kinetic_law.getFormula()
        # You can't keep the name max for the parameter because that throws an error in parsing here
        expression = parse_expr(formula.replace("max", cls.alternative_max_name))
        return expression, param_out_dict


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
