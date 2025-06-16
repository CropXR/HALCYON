import torch
from mlmmint.RegulatoryNetwork import RegulatoryNetwork
from mlmmint.Models.Model import Model
from mlmmint.Models.MechanisticModel import ComplexODEMechanisticModel
from sympy import symbols, sympify
import networkx as nx
import pytest
from mlmmint.Models.MechanisticModel import ComplexODEMechanisticModel

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
    return ComplexODEMechanisticModel(
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
