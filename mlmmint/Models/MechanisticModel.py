import numpy as np
from mlmmint.ExpressionData import ExpressionData
from mlmmint.GeneNetwork import GeneNetwork
from mlmmint.Models.BaseModel import BaseModel
from scipy.integrate import solve_ivp


class MechanisticModel(BaseModel):
    def __init__(self, network: GeneNetwork, expression_data: ExpressionData):
        self.network = network
        self.num_genes = len(network.genes)
        self.data = expression_data

    def filter_matching_genes(self):
        """When performed the genes in the expression data will be trimmed to only include"""
        self.data = self.data.keep_genes_in_list(self.network.genes)

    def fit(self):
        """
        Fit the model to expression data.

        Parameters:
        - data (ExpressionData): The gene expression data to fit the model to.
        """
        self.filter_matching_genes()
        # do actual fitting

    @property
    def gene_labels(self):
        return self.data.genes


class LinearMechanisticModel(MechanisticModel):
    def __init__(self, network: GeneNetwork, expression_data: ExpressionData):
        super().__init__(network, expression_data)

    def get_derivative(self, t, state: np.ndarray):
        """
        Defines the ODE system based on the gene interactions in the network.

        Parameters:
        - state (np.array): Current gene expression levels.
        - t (float): Time (not used here but required by odeint).

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
            self.get_derivative, t_span=timespan, y0=initial_state, t_eval=t_eval
        )
        return solution
