from pathlib import Path
import libsbml
import torch
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model
from halcyon.Models.MechanisticModel import ComplexODEModel, PyTorchODE
from sympy import symbols, sympify
import networkx as nx
import pytest
import pandas as pd
import numpy as np


def test_multiplicative_interactions_work():
    in_folder = Path("tests/4_genes_multplicative_interactions")
    sbml_path = in_folder / 'integration_test_network.xml'
    my_model = PyTorchODE.from_sbml(sbml_path)
    
    ss_values = pd.read_csv(in_folder / 'integration_test_network_nonoise_wildtype.tsv', sep='\t')
    ss_dict = ss_values.to_dict('records')[0]

    forward_results = my_model.forward(torch.tensor([0]), torch.tensor(list(ss_dict.values()), dtype=torch.float32))

    assert np.allclose(
        forward_results.detach().numpy(), np.array([0, 0, 0, 0 ]), atol=1e-5
    )
    # TODO then handle that this can handle multiple batches for integration over time
    output = my_model.integrate(torch.tensor(list(ss_dict.values()), dtype=torch.float32), torch.tensor([0., 0.5, 1.]))


# Define parameter dictionary
param_dict = {
    "pSPA": 1.0,
    "dSPA": 0.1,
    "assCOP1SPA1": 0.2,
    "dissCOP1SPA1": 0.05,
    "pCOP1": 1.0,
    "dCOP1": 0.1,
    "iCOP1_light": 0.1,
    "iCOP1_drought": 0.1,
    "iCOP1_heat_shoot": 1.0,
    "pCOP1_heat_shoot": 1.0,
    "iCOP1_SPA1_auto": 0.1,
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
        param_dict["pCOP1"] * param_dict["pCOP1_heat_shoot"] ** s["heat"]
        - param_dict["dCOP1"] * s["cop1_shoot"]
        - (
            param_dict["iCOP1_light"] * s["light"]
            + param_dict["iCOP1_drought"] * s["drought"]
        )
        * param_dict["iCOP1_heat_shoot"] ** s["heat"]
        * s["cop1_shoot"]
        - param_dict["assCOP1SPA1"] * s["cop1_shoot"] * s["spa1_shoot"]
        + param_dict["dissCOP1SPA1"] * s["cop1_spa1_shoot"]
    ),
    "cop1_spa1_shoot": lambda s: (
        param_dict["assCOP1SPA1"] * s["cop1_shoot"] * s["spa1_shoot"]
        - param_dict["dissCOP1SPA1"] * s["cop1_spa1_shoot"]
        - param_dict["iCOP1_SPA1_auto"] * s["cop1_spa1_shoot"] ** 2
    ),
}

input_dict = {
    "light": lambda t: (t % 24 <= 16),  # light on for 16 hours
    "heat": lambda t: 0,  # no heat stress
    "drought": lambda t: 0,  # no drought stress
}


# test complex ODE mechanistic model
@pytest.fixture
def complex_model():
    return ComplexODEModel(
        ODE_dict=ODE_dict,
        input_dict=input_dict,
        param_dict={
            k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
            for k, v in param_dict.items()
        },
    )


def test_forward(complex_model):
    t = torch.tensor(0.0)
    state = torch.tensor([1.0, 1.0, 1.0])
    derivatives = complex_model.forward(t, state)
    assert derivatives.shape == state.shape
    assert torch.is_tensor(derivatives)


def test_build_network_from_self(complex_model):
    net = complex_model._build_network_from_self()
    assert isinstance(net, nx.DiGraph)
    assert len(net.nodes) == len(ODE_dict) + len(input_dict)
    assert len(net.edges) > 0


def test_regulatory_network(complex_model):
    regulatory_network = complex_model.regulatory_network
    assert isinstance(regulatory_network, RegulatoryNetwork)
