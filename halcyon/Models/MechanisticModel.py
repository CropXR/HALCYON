import logging
import re
import libsbml
from pathlib import Path
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols
import torch
import numpy as np
from typing import Callable, Dict, List, Optional
from abc import abstractmethod
from typing_extensions import Self
from sympy.core.parameters import evaluate
from torch import nn
import networkx as nx

try:
    import sympytorch
except ImportError:  # pragma: no cover - handled at runtime
    sympytorch = None  # type: ignore

from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model


def _torch_heaviside(*args: torch.Tensor | float) -> torch.Tensor:
    if not args:
        raise ValueError("Heaviside requires at least one argument")

    x = args[0]
    x_tensor = x if torch.is_tensor(x) else torch.as_tensor(x)
    dtype = (
        x_tensor.dtype
        if x_tensor.dtype.is_floating_point
        else torch.get_default_dtype()
    )
    device = x_tensor.device if torch.is_tensor(x) else torch.device("cpu")

    if len(args) == 1:
        values = torch.full_like(x_tensor, 0.5, dtype=dtype)
    else:
        raw_values = args[1]
        if torch.is_tensor(raw_values):
            values = raw_values.to(device=device, dtype=dtype)
        else:
            values = torch.full_like(x_tensor, float(raw_values), dtype=dtype)

    return torch.heaviside(x_tensor.to(device=device, dtype=dtype), values)


_EXTRA_SYMPY_FUNCS: Dict[sp.FunctionClass, Callable[..., torch.Tensor]] = {
    sp.Heaviside: _torch_heaviside,
}


class MechanisticModel(Model):
    def __init__(self, modelled_gene_labels: List[str], condition_names: List[str]):
        super().__init__(modelled_gene_labels, condition_names)

    # @property
    # @abstractmethod
    # def regulatory_network(self) -> RegulatoryNetwork:
    #     """Return the regulatory network of the model."""
    #     pass


class PyTorchODE(MechanisticModel):
    alternative_max_name = "max_rate"

    def __init__(
        self,
        modelled_gene_labels,
        condition_names,
        synthesis_dict,
        decay_dict,
        param_dict,
    ):
        super().__init__(modelled_gene_labels, condition_names)
        self.synthesis_dict = synthesis_dict
        self.decay_dict = decay_dict
        self.param_dict = param_dict
        self._compiled_ready = False
        self._synthesis_module: Optional[nn.Module] = None
        self._decay_module: Optional[nn.Module] = None

    def select_genes(self, genes: list[str]):
        assert all(gene in self.modelled_gene_labels for gene in genes)

    @classmethod
    def from_sbml(
        cls, sbml_path, condition_names: List[str], from_GNW: bool = False
    ) -> Self:

        if isinstance(sbml_path, Path):
            sbml_path = str(sbml_path)
        assert isinstance(sbml_path, str)  # Ensure sbml_path is a string
        print("Loading SBML model from:", sbml_path)

        reader = libsbml.SBMLReader()
        document = reader.readSBML(sbml_path)

        # assert document is not None and document.getModel() is not None, "Failed to load SBML model."
        model = document.getModel()

        param_dict = dict()
        decay_expression_dict = dict()
        synthesis_expression_dict = dict()
        base_node_names = []

        for i in range(model.getNumReactions()):
            reaction = model.getReaction(i)
            reaction_id = reaction.getId()  # .rstrip("_")
            formula = reaction.getKineticLaw().getFormula()

            if from_GNW:
                formula = cls._fix_gnw_division_brackets(formula)

            libsbml_parameters = (
                model.getReaction(reaction_id).getKineticLaw().getListOfParameters()
            )

            base_node_name, sympy_formula, one_reactions_parameters = (
                cls.parse_one_reaction(reaction_id, formula, libsbml_parameters)
            )
            if base_node_name not in base_node_names:
                base_node_names.append(base_node_name)

            assert set(param_dict.keys()).isdisjoint(
                one_reactions_parameters.keys()
            ), "Parameter duplicates"
            param_dict.update(one_reactions_parameters)
            if reaction_id.endswith("_degradation"):
                # print("Adding degradation for", base_node_name, "formula:", sympy_formula)
                decay_expression_dict[base_node_name] = sympy_formula
            elif reaction_id.endswith("_synthesis"):
                # print("Adding synthesis for", base_node_name, "formula:", sympy_formula)
                synthesis_expression_dict[base_node_name] = sympy_formula
            else:
                raise NameError(
                    f"Reaction {reaction_id} does not end with _synthesis or _degradation."
                )
        # print("Synthesis expressions:", synthesis_expression_dict)
        # print("Decay expressions:", decay_expression_dict)

        return cls(
            base_node_names,
            condition_names,
            synthesis_expression_dict,
            decay_expression_dict,
            param_dict,
        )

    def _compile_expressions(self) -> None:
        """Pre-compile SymPy expressions into torch modules via sympytorch."""
        if (
            sympytorch is None
        ):  # pragma: no cover - executed only when missing dependency
            raise ImportError(
                "sympytorch is required for MechanisticModel. Install it via 'pip install sympytorch'."
            )

        param_subs = {symbols(name): value for name, value in self.param_dict.items()}

        def _prepare_expressions(expr_dict: Dict[str, sp.Expr]) -> List[sp.Expr]:
            prepared: List[sp.Expr] = []
            for gene in self.modelled_gene_labels:
                expr = expr_dict[gene].subs(param_subs)
                expr = sympytorch.hide_floats(expr)
                prepared.append(expr)
            return prepared

        synthesis_exprs = _prepare_expressions(self.synthesis_dict)
        decay_exprs = _prepare_expressions(self.decay_dict)

        try:
            self._synthesis_module = sympytorch.SymPyModule(
                expressions=synthesis_exprs,
                extra_funcs=_EXTRA_SYMPY_FUNCS,
            )
            self._decay_module = sympytorch.SymPyModule(
                expressions=decay_exprs,
                extra_funcs=_EXTRA_SYMPY_FUNCS,
            )
        except KeyError as err:
            func_name = (
                getattr(err.args[0], "__name__", str(err.args[0]))
                if err.args
                else "unknown"
            )
            raise ValueError(
                f"Unsupported sympy function '{func_name}' in SBML expression. "
                "Provide a torch implementation via _EXTRA_SYMPY_FUNCS or extend sympytorch."
            ) from err

        self._compiled_ready = True

    @classmethod
    def parse_one_reaction(
        cls,
        name: str,
        formula_string: str,
        libsbml_parameters: libsbml.ListOfParameters,
    ):
        base_node_name = name.replace("_degradation", "").replace("_synthesis", "")
        formula_string = re.sub(r"\bmax\b", cls.alternative_max_name, formula_string)

        param_dict = dict()

        # This way the order of the terms is preserved, eases readability
        with evaluate(0):
            # TODO THIJS: here the formula string for Node_1_synthesis needs the brackets added just after the division. Ideally check if this method would also generalise for three or more genes.
            sympy_formula = parse_expr(formula_string, evaluate=False)

        replacement_dict = dict()

        for old_param_name in sympy_formula.free_symbols:
            old_param_name = str(old_param_name)
            if old_param_name.startswith("Node"):
                continue
            elif old_param_name == "cell":
                continue
            new_param_name = f"{old_param_name}_{base_node_name}"
            replacement_dict[old_param_name] = new_param_name
            # Also figure out the value of the parameter
            assert (
                new_param_name not in param_dict
            ), f"Parameter {new_param_name} already exists in param_dict. Check for naming conflicts."
            if old_param_name == cls.alternative_max_name:
                old_param_name = "max"
            param_dict[new_param_name] = libsbml_parameters.get(
                old_param_name
            ).getValue()
        sympy_formula = sympy_formula.subs(replacement_dict)

        return base_node_name, sympy_formula, param_dict

    @classmethod
    def _fix_gnw_division_brackets(cls, formula: str) -> str:
        """
        Fix GNW division parsing by properly grouping into brackets all multiplicative terms
        that should be in the denominator after a division operator.
        """
        import re

        def find_matching_paren(s, start_pos):
            """Find the position of the matching closing parenthesis"""
            if s[start_pos] != "(":
                return -1

            count = 1
            pos = start_pos + 1

            while pos < len(s) and count > 0:
                if s[pos] == "(":
                    count += 1
                elif s[pos] == ")":
                    count -= 1
                pos += 1

            return pos - 1 if count == 0 else -1

        def fix_division_at_position(formula_str, div_pos):
            """Fix division at a specific position by grouping subsequent multiplication terms"""
            if div_pos >= len(formula_str) - 1:
                return formula_str

            # Start collecting terms after the division
            pos = div_pos + 1
            terms = []

            while pos < len(formula_str):
                # Skip whitespace
                while pos < len(formula_str) and formula_str[pos].isspace():
                    pos += 1

                if pos >= len(formula_str):
                    break

                # If this is not the first term, we need to check for multiplication operator
                if len(terms) > 0:
                    if formula_str[pos] == "*":
                        pos += 1  # Skip the '*'
                        # Skip whitespace after '*'
                        while pos < len(formula_str) and formula_str[pos].isspace():
                            pos += 1
                    else:
                        # No multiplication operator, so we're done with this division
                        break

                if pos >= len(formula_str):
                    break

                # Collect the next term
                term_start = pos

                if formula_str[pos] == "(":
                    # Parenthetical expression
                    end_paren = find_matching_paren(formula_str, pos)
                    if end_paren != -1:
                        terms.append(formula_str[pos : end_paren + 1])
                        pos = end_paren + 1
                    else:
                        break
                else:
                    # Regular term (variable, number, or function call)
                    while (
                        pos < len(formula_str)
                        and formula_str[pos] not in ["*", "/", "+", "-", ")", " "]
                        and not (formula_str[pos : pos + 1].isspace())
                    ):
                        pos += 1

                    # Handle function calls like pow(...)
                    if pos < len(formula_str) and formula_str[pos] == "(":
                        end_paren = find_matching_paren(formula_str, pos)
                        if end_paren != -1:
                            pos = end_paren + 1

                    terms.append(formula_str[term_start:pos])

            # If we found multiple terms, wrap them in parentheses
            if len(terms) > 1:
                grouped_terms = " * ".join(terms)  # Preserve spacing
                new_denominator = f"/({grouped_terms})"

                # Calculate the original length to replace
                original_end = pos
                original_section = formula_str[div_pos:original_end]

                return (
                    formula_str[:div_pos] + new_denominator + formula_str[original_end:]
                )

            return formula_str

        # Find all division operators and fix them
        result = formula
        div_positions = [i for i, char in enumerate(result) if char == "/"]

        # Process divisions from right to left to avoid position shifts
        for div_pos in reversed(div_positions):
            result = fix_division_at_position(result, div_pos)

        return result

    def integrate(
        self,
        initial_state: torch.Tensor,
        t_eval: torch.Tensor,
    ) -> torch.Tensor:
        self._synthesis_module = None
        self._decay_module = None
        self._compiled_ready = False
        self._compile_expressions()
        return super().integrate(initial_state, t_eval)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Get the derivative at a given time point.
        Parameters:
            t (torch.Tensor): The time point at which to evaluate the ODE.
            state (torch.Tensor): The current state of the system.
            impute_cell_to_1 (bool): Whether to impute the 'cell' variable to 1. Default is True.
        Returns:
            torch.Tensor: The derivative of the state at the given time point in the shape [Batch, Genes].
        """
        if state.isinf().any() or state.isnan().any():
            raise ValueError("NaN or inf in state tensor.")

        # Clamp state to prevent negative values
        state = torch.clamp(state, min=0.0)
        if not getattr(self, "_compiled_ready", False):
            self._compile_expressions()
        batch_size, n_genes = state.shape
        device = state.device

        synthesis_module = self._synthesis_module
        decay_module = self._decay_module
        if synthesis_module is None or decay_module is None:
            raise RuntimeError("SymPy modules were not initialised correctly.")

        synthesis_module.to(device=device, dtype=state.dtype)
        decay_module.to(device=device, dtype=state.dtype)

        ko_masks = getattr(self, "ko_masks", torch.ones_like(state))
        if ko_masks.dim() == 1:
            if ko_masks.shape[0] == n_genes:
                ko_masks = ko_masks.unsqueeze(0).expand(batch_size, n_genes)
            elif ko_masks.shape[0] == batch_size:
                ko_masks = ko_masks.unsqueeze(1).expand(batch_size, n_genes)
            else:
                raise ValueError(
                    f"ko_masks shape {ko_masks.shape} does not match state shape {state.shape}"
                )
        elif ko_masks.shape != state.shape:
            ko_masks = ko_masks.expand_as(state)
        ko_masks = ko_masks.to(device=device, dtype=state.dtype)

        inputs = {
            gene: state[:, idx] for idx, gene in enumerate(self.modelled_gene_labels)
        }
        inputs["cell"] = torch.full(
            (batch_size,),
            1.0,
            dtype=state.dtype,
            device=device,
        )

        synthesis = synthesis_module(**inputs)
        decay = decay_module(**inputs)

        target_shape = (batch_size, n_genes)
        try:
            synthesis = torch.broadcast_to(synthesis, target_shape)
            decay = torch.broadcast_to(decay, target_shape)
        except RuntimeError as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(
                f"SymPyModule outputs have incompatible shapes {synthesis.shape} and {decay.shape}."
            ) from exc

        return synthesis * ko_masks - decay
