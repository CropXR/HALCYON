import numpy as np
import torch
from mlmmint.DataHandlers import GeneExpressionDataHandler
from mlmmint.ExpressionData import ExpressionData
from mlmmint.GeneNetwork import GeneNetwork
from mlmmint.Models.BaseModel import BaseModel
from scipy.integrate import solve_ivp

from torchdiffeq import odeint_adjoint as odeint
from torch import nn


class MechanisticModel(BaseModel):
    def __init__(self, network: GeneNetwork, expression_data: ExpressionData):
        self.network = network
        self.num_genes = len(network.genes)
        self.data_handler = GeneExpressionDataHandler(expression_data, network)
        self.data_handler.filter_matching_genes()

    def fit(self):
        """Fit the model to expression data."""
        # Now we can proceed with model fitting using filtered_data
        pass

    @property
    def gene_labels(self):
        return self.data_handler.get_genes()


class LinearMechanisticModel(MechanisticModel):
    def __init__(self, network: GeneNetwork, expression_data: ExpressionData):
        super().__init__(network, expression_data)

    def forward(self, t, state: np.ndarray):
        """
        Defines the ODE system based on the gene interactions in the network.

        Parameters:
        - t (float): Time (not used here but required by odeint).
        - state (np.array): Current gene expression levels.

        Returns:
        - dydt (np.array): The rate of change of gene expression levels.
        """
        dydt = np.zeros(
            self.num_genes
        )  # Array to store the rate of change of each gene

        # For each gene, calculate its rate of change as a sum of influences
        for i, gene in enumerate(self.gene_labels):
            influence_sum = 0

            # Sum the influences from all other genes (edges in the network)
            for (
                from_gene,
                to_gene,
                property_dict,
            ) in self.network.get_ingoing_interactions(gene):
                if to_gene == gene:
                    # Find the index of the genes in the list
                    from_index = self.gene_labels.index(from_gene)
                    to_index = self.gene_labels.index(to_gene)

                    # Update the influence sum (apply the weight of the influence)
                    influence_sum += property_dict["influence"] * state[from_index]

            # Update the rate of change for the gene
            dydt[i] = influence_sum

        return dydt

    def integrate(self, initial_state: np.ndarray, timespan: list, t_eval=None):
        """
        Simulate the gene expression dynamics over time.

        Parameters:
        - time_points (np.array): Time points at which to evaluate the gene expressions.

        Returns:
        - solution (np.array): The gene expression levels at each time point.
        """
        solution = solve_ivp(
            self.forward, t_span=timespan, y0=initial_state, t_eval=t_eval
        )
        return solution


class DerivativeFunction(nn.Module):
    def __init__(self, weight_matrix):
        super().__init__()
        self.weights = nn.Parameter(
            weight_matrix
        )  # Keeps it as a trainable parameter if needed

    def forward(self, t, state):
        return torch.matmul(self.weights, state)


class TorchLinearODE(MechanisticModel):
    def __init__(self, network: GeneNetwork, expression_data: ExpressionData):
        super().__init__(network, expression_data)

        # Create a weight matrix for the network
        self.gene_index = {gene: i for i, gene in enumerate(self.gene_labels)}
        self.num_genes = len(self.gene_labels)

        # Initialize the adjacency/weight matrix
        weight_matrix = torch.zeros(
            (self.num_genes, self.num_genes), dtype=torch.float32
        )

        for from_gene, to_gene, property_dict in network.get_all_interactions():
            if from_gene in self.gene_index and to_gene in self.gene_index:
                from_idx = self.gene_index[from_gene]
                to_idx = self.gene_index[to_gene]
                weight_matrix[to_idx, from_idx] = property_dict["influence"]

                # Store the derivative function as an nn.Module
        self.derivative_fn = DerivativeFunction(weight_matrix)

    def forward(self, state: torch.Tensor, t):
        return self.derivative_fn

    def integrate(self, initial_state, t_eval):
        output: torch.Tensor = odeint(self.derivative_fn, initial_state, t_eval)  # type: ignore
        # Transposed for some reason
        return output.T

    def fit(self, data: ExpressionData):
        pass
