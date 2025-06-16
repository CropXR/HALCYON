import torch
from mlmmint.RegulatoryNetwork import RegulatoryNetwork
from mlmmint.Models.Model import Model
from mlmmint.Models.MechanisticModel import ComplexODEMechanisticModel
from sympy import symbols, sympify
import networkx as nx

# Define parameter dictionary
param_dict = {
    "pSPA": 1.0, "dSPA": 0.1, "assCOP1SPA1": 0.2, "dissCOP1SPA1": 0.05,
    "pCOP1": 1.0, "dCOP1": 0.1, "iCOP1_light": 0.1, "iCOP1_drought": 0.1,
    "iCOP1_heat_shoot": 1.0, "pCOP1_heat_shoot": 1.0, "iCOP1_SPA1_auto": 0.1,
    # Add any other missing parameters here
}

# Define ODEs as lambdas that take the full state‐dict
ODE_dict = {
    "spa1_shoot": lambda s: (
        param_dict["pSPA"]
        - param_dict["dSPA"] * s["spa1_shoot"]
        - param_dict["assCOP1SPA1"] * s["cop1_shoot"] * s["spa1_shoot"]
        + param_dict["dissCOP1SPA1"] * s["cop1_spa1_shoot"]
    ),
    "cop1_shoot": lambda s: (
        param_dict["pCOP1"] * param_dict["pCOP1_heat_shoot"]**s["heat"]
        - param_dict["dCOP1"] * s["cop1_shoot"]
        - (param_dict["iCOP1_light"]*s["light"] + param_dict["iCOP1_drought"]*s["drought"])
          * param_dict["iCOP1_heat_shoot"]**s["heat"] * s["cop1_shoot"]
        - param_dict["assCOP1SPA1"] * s["cop1_shoot"] * s["spa1_shoot"]
        + param_dict["dissCOP1SPA1"] * s["cop1_spa1_shoot"]
    ),
    "cop1_spa1_shoot": lambda s: (
        param_dict["assCOP1SPA1"] * s["cop1_shoot"] * s["spa1_shoot"]
        - param_dict["dissCOP1SPA1"] * s["cop1_spa1_shoot"]
        - param_dict["iCOP1_SPA1_auto"] * s["cop1_spa1_shoot"]**2
    ),
    # … define hy5_shoot, hy5p_shoot, spa1_root, … etc. …
}

# Define the regulatory network
# Get a list of the dependencies of each equation
# dependencies = {
#     "spa1_shoot":   ["cop1_shoot", "cop1_spa1_shoot"],
#     "cop1_shoot":   ["heat", "light", "drought", "spa1_shoot", "cop1_spa1_shoot"],
#     "cop1_spa1_shoot": ["cop1_shoot", "spa1_shoot"],
# }
dependencies = {}
gene_names = list(ODE_dict.keys()) # must be consistent
for var, ODE in ODE_dict.items():
    # Convert the lambda function to a symbolic expression
    symbolic_vars = symbols(gene_names + ["light", "heat", "drought"])
    symbolic_expr = sympify(ODE({v: symbolic_vars[i] for i, v in enumerate(gene_names)} | 
                                   {"light": symbolic_vars[-3], "heat": symbolic_vars[-2], "drought": symbolic_vars[-1]}))
    
    # Extract free symbols (dependencies) from the symbolic expression
    dependencies[var] = [str(sym) for sym in symbolic_expr.free_symbols if str(sym) in gene_names or str(sym) in ("light", "heat", "drought")]

# Build the directed graph
nx_net = nx.DiGraph()
# Add nodes for each variable with their respective ODE functions as data
for var, ODE in ODE_dict.items():
    nx_net.add_node(var, ODE=ODE, type="network")

# Add input nodes (light, heat, drought) with their respective time series functions as data
for input_node, time_series_func in zip(
    ["light", "heat", "drought"], 
    [lambda t: (t % 24 <= 16), lambda t: 0, lambda t: 0] #time series functions for light, heat, and drought
):
    nx_net.add_node(input_node, time_series_func=time_series_func, type="input")

# Create edges from each dependency → dependent variable
for var, deps in dependencies.items():
    for dep in deps:
        nx_net.add_edge(dep, var)

# Create the regulatory network and Mechanistic model from the graph
mechanistic_model = ComplexODEMechanisticModel(RegulatoryNetwork(nx_net), param_dict)