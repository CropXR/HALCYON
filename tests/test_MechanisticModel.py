import numpy as np
import torch
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model
from halcyon.Models.MechanisticModel import ComplexODEModel
from sympy import symbols, sympify
import networkx as nx
import pytest
from halcyon.Models.MechanisticModel import ComplexODEModel


# test complex ODE mechanistic model
@pytest.fixture
def complex_model():
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
    ode_dict = {
        "spa1_shoot": lambda state: (
            param_dict["pSPA"]
            - param_dict["dSPA"] * state["spa1_shoot"]
            - param_dict["assCOP1SPA1"] * state["cop1_shoot"] * state["spa1_shoot"]
            + param_dict["dissCOP1SPA1"] * state["cop1_spa1_shoot"]
        ),
        "cop1_shoot": lambda state: (
            param_dict["pCOP1"] * param_dict["pCOP1_heat_shoot"] ** state["heat"]
            - param_dict["dCOP1"] * state["cop1_shoot"]
            - (
                param_dict["iCOP1_light"] * state["light"]
                + param_dict["iCOP1_drought"] * state["drought"]
            )
            * param_dict["iCOP1_heat_shoot"] ** state["heat"]
            * state["cop1_shoot"]
            - param_dict["assCOP1SPA1"] * state["cop1_shoot"] * state["spa1_shoot"]
            + param_dict["dissCOP1SPA1"] * state["cop1_spa1_shoot"]
        ),
        "cop1_spa1_shoot": lambda state: (
            param_dict["assCOP1SPA1"] * state["cop1_shoot"] * state["spa1_shoot"]
            - param_dict["dissCOP1SPA1"] * state["cop1_spa1_shoot"]
            - param_dict["iCOP1_SPA1_auto"] * state["cop1_spa1_shoot"] ** 2
        ),
    }

    # TODO: Think about way of merging this with the ExperimentCondition class
    # Would make it compatible with the Trainer class, which needs (ExpressionDataWithConditions). Also allows us to test if the parameters can be trained.
    # TODO: Bit lost here; the ode_dict never takes T as an input, but the ComplexODEModel does have a forward method that takes t as an input.
    # TODO might be good to think about how to handle knockouts here as well
    input_dict = {
        "light": lambda t: (t % 24 <= 16),  # light on for 16 hours
        "heat": lambda t: 0,  # no heat stress
        "drought": lambda t: 0,  # no drought stress
    }

    return ComplexODEModel(
        ode_dict=ode_dict,
        input_dict=input_dict,
        param_dict=param_dict,
    )


def test_forward(complex_model):
    t = torch.tensor(0.0)
    state = torch.tensor([1.0, 1.0, 1.0])
    derivatives = complex_model.forward(t, state)
    assert len(derivatives) == len(complex_model.modelled_gene_labels)
    assert np.allclose(derivatives, [0.7500, 0.6500, 0.0500])

    state = torch.tensor([0.0, 0.0, 0.0])
    derivatives = complex_model.forward(t, state)
    assert np.allclose(derivatives, [1, 1, 0])

    state = torch.tensor([1, 1, 1])
    derivatives = complex_model.forward(torch.tensor(18.0), state)
    assert np.allclose(derivatives, [0, 0, 0])


def test_integrate(complex_model):
    output = complex_model.integrate(
        torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([0.0, 1.0, 2.0, 3.0])
    )
    # Still broken because integration returns a tensor not a dict I think
    # TODO: Idea put the results of a correct integration run you ran yourself here so you have something to compare it to.


def test_regulatory_network(complex_model):
    regulatory_network = complex_model.get_regulatory_network()
    assert regulatory_network.get_all_interactions(with_data=False) == [
        ("spa1_shoot", "spa1_shoot"),
        ("spa1_shoot", "cop1_shoot"),
        ("spa1_shoot", "cop1_spa1_shoot"),
        ("cop1_shoot", "spa1_shoot"),
        ("cop1_shoot", "cop1_shoot"),
        ("cop1_shoot", "cop1_spa1_shoot"),
        ("cop1_spa1_shoot", "spa1_shoot"),
        ("cop1_spa1_shoot", "cop1_shoot"),
        ("cop1_spa1_shoot", "cop1_spa1_shoot"),
        ("light", "cop1_shoot"),
        ("heat", "cop1_shoot"),
        ("drought", "cop1_shoot"),
    ]
    assert tuple(regulatory_network.nodes) == (
        "spa1_shoot",
        "cop1_shoot",
        "cop1_spa1_shoot",
        "light",
        "heat",
        "drought",
    )
