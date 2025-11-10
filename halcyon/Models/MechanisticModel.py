import logging
import re
import libsbml
from pathlib import Path
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols
import torch
from typing import Callable, Dict, List, Optional
from abc import abstractmethod
from typing_extensions import Self
from sympy.core.parameters import evaluate
from torch import nn
import networkx as nx
import sympytorch
import logging

from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model


class MechanisticModel(Model):
    def __init__(self, modelled_gene_labels: List[str], condition_names: List[str]):
        super().__init__(modelled_gene_labels, condition_names)

    @property
    @abstractmethod
    def regulatory_network(self) -> RegulatoryNetwork:
        """Return the regulatory network of the model."""
        pass


class PyTorchODE(MechanisticModel):
    alternative_max_name = "max_rate"

    def __init__(
        self,
        modelled_gene_labels: List[str],
        condition_names: list[str],
        synthesis_dict: Dict[str, sp.Expr],
        decay_dict: Dict[str, sp.Expr],
        param_dict: Dict[str, float],
    ):
        super().__init__(modelled_gene_labels, condition_names)
        self.synthesis_dict = synthesis_dict
        self.decay_dict = decay_dict
        # Convert param_dict to a trainable ParameterDict
        self.param_dict = nn.ParameterDict(
            {
                k: nn.Parameter(
                    torch.tensor(v, dtype=torch.float32), requires_grad=True
                )
                for k, v in param_dict.items()
            }
        )
        self._compiled_ready = False  # should be false until expressions are compiled, which is done at first integration
        self._synthesis_module: Optional[nn.Module] = (
            None  # compiled synthesis modules will be stored here after compilation
        )
        self._decay_module: Optional[nn.Module] = (
            None  # compiled decay modules will be stored here after compilation
        )

    @property
    def regulatory_network(self) -> RegulatoryNetwork:
        raise NotImplementedError("This method needs to be implemented in subclass.")

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
        Returns:
            torch.Tensor: The derivative of the state at the given time point in the shape [Batch, Genes].
        """
        if state.isinf().any() or state.isnan().any():
            raise ValueError("NaN or inf in state tensor.")

        # Clamp state to prevent negative values, for some reason without this the compiled modules can produce NaNs/complex values from negative inputs
        # most likely, the steps are slightly too large and lead to negative concentrations
        state = torch.clamp(state, min=0.0)
        if not self._compiled_ready:
            self._compile_expressions()
        batch_size, n_genes = state.shape

        if self._synthesis_module is None or self._decay_module is None:
            raise RuntimeError("SymPy modules were not initialised correctly.")

        self.ko_masks = self.ko_masks.to(device=self.device)

        inputs = {
            gene: state[:, idx] for idx, gene in enumerate(self.modelled_gene_labels)
        }
        # cell is a required variable in SBML expressions, it indicates the cell volume used to scale ODE's, we set it to 1
        inputs["cell"] = torch.full(
            (batch_size,),
            1.0,
            dtype=state.dtype,
            device=self.device,
        )

        synthesis = self._synthesis_module(**inputs)
        decay = self._decay_module(**inputs)
        # If output is 1D (single sample), unsqueeze to add batch dimension
        if synthesis.dim() == 1:
            synthesis = synthesis.unsqueeze(0)
        if decay.dim() == 1:
            decay = decay.unsqueeze(0)

        return synthesis * self.ko_masks - decay

    def set_parameters(self, param_values: dict):
        """
        Set parameter values in the model. Accepts a dict of {name: value} and updates the corresponding nn.Parameter in param_dict.
        Useful for testing and manual parameter updates.
        """
        for k, v in param_values.items():
            if k in self.param_dict:
                self.param_dict[k].data = torch.tensor(
                    v, dtype=self.param_dict[k].dtype, device=self.param_dict[k].device
                )
        self._compiled_ready = False  # recompile expressions with new parameter values

    def select_genes(self, genes_to_keep: list[str]):
        assert all(
            gene_to_keep in self.modelled_gene_labels for gene_to_keep in genes_to_keep
        ), "Not all genes that need to be kept are in the model"
        genes_to_remove = list(set(self.modelled_gene_labels) ^ set(genes_to_keep))

        new_synthesis_dict = dict()
        new_decay_dict = dict()

        # Heuristic, set all gene expressions to 0 for genes that are not modelled
        replacement_dict = {unmodeled_gene: 0 for unmodeled_gene in genes_to_remove}
        for gene in genes_to_keep:
            synthesis_formula = self.synthesis_dict[gene]
            new_synthesis_dict[gene] = synthesis_formula.subs(replacement_dict)
            new_decay_dict[gene] = self.decay_dict[gene]
        self._compiled_ready = False
        self.synthesis_dict = new_synthesis_dict
        self.decay_dict = new_decay_dict
        self.modelled_gene_labels = genes_to_keep

    @classmethod
    def from_sbml(
        cls, sbml_path: str | Path, condition_names: list[str], from_gnw: bool = False
    ) -> Self:
        """
        Create a MechanisticModel from an SBML file.

        Args:
            sbml_path (string or path): The path to the SBML file.
            from_gnw (bool, optional): Should be true when the model file to be read is generated by genenetweaver, as missing brackets in loaded GNW files are added when True. Defaults to False.

        Raises:
            NameError: If the SBML model is invalid or cannot be loaded.

        Returns:
            Self: An instance of the MechanisticModel.
        """

        if isinstance(sbml_path, Path):
            sbml_path = str(sbml_path)
        assert isinstance(sbml_path, str)  # Ensure sbml_path is a string
        logging.info(f"Loading SBML model from: {sbml_path}")

        reader = libsbml.SBMLReader()
        document = reader.readSBML(sbml_path)

        assert (
            document is not None and document.getModel() is not None
        ), "Failed to load SBML model."
        model = document.getModel()

        param_dict = dict()
        decay_expression_dict = dict()
        synthesis_expression_dict = dict()
        base_node_names = []

        for i in range(model.getNumReactions()):
            reaction = model.getReaction(i)
            reaction_id = reaction.getId()
            formula = reaction.getKineticLaw().getFormula()

            if from_gnw:
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
                decay_expression_dict[base_node_name] = sympy_formula
            elif reaction_id.endswith("_synthesis"):
                synthesis_expression_dict[base_node_name] = sympy_formula
            else:
                raise NameError(
                    f"Reaction {reaction_id} does not end with _synthesis or _degradation."
                )
        return cls(
            base_node_names,
            condition_names,
            synthesis_expression_dict,
            decay_expression_dict,
            param_dict,
        )

    @classmethod
    def parse_one_reaction(
        cls,
        name: str,
        formula_string: str,
        libsbml_parameters: libsbml.ListOfParameters,
    ):
        """

        Parse a single reaction from the SBML model.
        Args:
            name (str): The name of the reaction.
            formula_string (str): The formula string of the reaction.
            libsbml_parameters (libsbml.ListOfParameters): The list of parameters associated with the reaction.
        Returns:
            tuple[str, sp.Expr, Dict[str, float]]: A tuple containing the base node name, the sympy expression, and a dictionary of parameters used in the equation.
        """
        base_node_name = name.replace("_degradation", "").replace("_synthesis", "")
        formula_string = re.sub(r"\bmax\b", cls.alternative_max_name, formula_string)

        param_dict = dict()

        # This way the order of the terms is preserved, eases readability
        with evaluate(0):
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

    def _set_all_parameters_to_zero(self):
        for key in self.param_dict.keys():
            self.param_dict[key].data.fill_(0.0)
        self._compiled_ready = False  # recompile expressions with new parameter values

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
            """Prepare SymPy expressions for compilation.
            Takes a dictionary of expressions and applies parameter substitutions from the parameter dictionary as well as float hiding. This ensures this is only done once per expression before compilation and not at each forward pass.
            Float hiding prevents SymPy from automatically evaluating or simplifying expressions that contain floats, which can be important for preserving the exact structure of symbolic expressions

            Args:
                expr_dict (Dict[str, sp.Expr]): A dictionary expression name keys and SymPy expressions values.

            Returns:
                List[sp.Expr]: A list of prepared SymPy expressions.
            """
            prepared: List[sp.Expr] = []
            for gene in self.modelled_gene_labels:
                expr = expr_dict[gene].subs(
                    param_subs
                )  # Apply parameter substitutions, take parameter values from param_dict
                expr = sympytorch.hide_floats(expr)
                prepared.append(expr)
            return prepared

        synthesis_exprs = _prepare_expressions(self.synthesis_dict)
        decay_exprs = _prepare_expressions(self.decay_dict)

        try:
            self._synthesis_module = sympytorch.SymPyModule(expressions=synthesis_exprs)
            self._decay_module = sympytorch.SymPyModule(expressions=decay_exprs)
        except KeyError as err:
            # extracting the function name from an error object in a defensive way, in case args is empty or does not have the expected structure
            func_name = (
                getattr(err.args[0], "__name__", str(err.args[0]))
                if err.args
                else "unknown"
            )
            raise ValueError(
                f"Unsupported sympy function '{func_name}' in SBML expression. "
                "Provide a torch implementation via sympytorch."
            ) from err

        self._synthesis_module.to(device=self.device)
        self._decay_module.to(device=self.device)
        self._compiled_ready = True

    @classmethod
    def _fix_gnw_division_brackets(cls, formula: str) -> str:
        """
        Fix incorrect division operator precedence by adding parentheses in GeneNetWeaver's SBML output.

        Somewhere in GeneNetWeaver exporting its model equations to SBML and libSBML importing them there is a bug:
        division expressions are written with incorrect operator precedence. For example, "a/b*c" is
        intended to mean "a/(b*c)" but SymPy would parse it as "(a/b)*c" due to left-to-right evaluation.

        This method scans the formula string for division operators and wraps all subsequent
        multiplication terms in parentheses to ensure they are treated as part of the denominator.

        Examples:
            "a/b*c" → "a/(b * c)"
            "a/b*c + d/e*f*g" → "a/(b * c)+ d/(e * f * g)"
            "x/pow(y,2)*z" → "x/(pow(y,2) * z)"

        Args:
            formula (str): The formula string from the SBML file, potentially with incorrect precedence.

        Returns:
            str: The corrected formula with proper parenthesization.

        Note:
            This function must be called when loading SBML files from GeneNetWeaver.
            Use `PyTorchODE.from_sbml(path, from_gnw=True)` to automatically apply this fix.
        """

        def _find_matching_closing_parenthesis(formula_str: str, start_pos: int) -> int:
            """
            Find the position of the matching closing parenthesis for an opening parenthesis.

            Args:
                formula_str (str): The string to search in.
                start_pos (int): The index of the opening parenthesis.

            Returns:
                int: The index of the matching closing parenthesis, or -1 if not found or
                     if the character at start_pos is not an opening parenthesis.
            """
            if formula_str[start_pos] != "(":
                return -1

            count = 1
            pos = start_pos + 1

            while pos < len(formula_str) and count > 0:
                if formula_str[pos] == "(":
                    count += 1
                elif formula_str[pos] == ")":
                    count -= 1
                pos += 1

            return pos - 1 if count == 0 else -1

        def _fix_division_at_position(formula_str: str, div_pos: int) -> str:
            """
            Fix a division operator at a specific position by grouping subsequent multiplication terms.

            Starting from the division operator at div_pos, this function scans forward to identify
            all terms that should be part of the denominator. Terms are separated by multiplication
            operators ('*'). The scanning stops when a different operator (+, -, /) is encountered
            or the end of the string is reached.

            The function handles:
            - Simple variables and numbers: "a", "123", "k_1"
            - Parenthetical expressions: "(x + y)"
            - Function calls: "pow(x, 2)", "max(a, b)"
            - Nested parentheses: "pow((x + y), 2)"

            Args:
                formula_str (str): The complete formula string.
                div_pos (int): The index of the division operator to fix.

            Returns:
                str: The formula with the division at div_pos corrected.
                     If only one term follows the division, no changes are made.
                     If multiple terms follow (connected by '*'), they are wrapped in parentheses.

            Example:
                For "a/b*c*d + e", with div_pos pointing to '/', this returns "a/(b * c * d) + e"
            """
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
                    end_paren = _find_matching_closing_parenthesis(formula_str, pos)
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
                        end_paren = _find_matching_closing_parenthesis(formula_str, pos)
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
            result = _fix_division_at_position(result, div_pos)

        return result
