import torch
from halcyon.Models import PhoenixOnlyAddLayers
import pytest
from torch.testing import assert_close


@pytest.fixture
def neural_ode():
    gene_labels = ["a", "b", "c"]
    my_neural_ode = PhoenixOnlyAddLayers(gene_labels, ("test_cond",), hidden_size=2)
    return my_neural_ode


@pytest.fixture
def init_values():
    return torch.tensor(
        [
            1.0,
            2.0,
            3.0,
        ]
    ).unsqueeze(0)


def test_create_neuralODE(neural_ode, init_values):
    neural_ode.set_conditions(torch.tensor([[0.0] * 4]))
    t = torch.tensor([0.0, 0.5, 1.0, 1.5])
    output = neural_ode.integrate(init_values, t)
    assert output.shape == (1, 3, 4)


def test_ko_neural_ode(neural_ode, init_values):
    # Knockout gene
    neural_ode.set_conditions(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    forward_pass_with_ko = neural_ode.forward(torch.tensor([0]), init_values)
    assert forward_pass_with_ko[0, 1] < 0
