import torch
from halcyon.Models import PhoenixOnlyAddLayers, MLP, CellBox
import pytest
from torch.testing import assert_close


@pytest.fixture(params=["phoenix", "mlp", "cellbox"])
def ode_model(request):
    gene_labels = ["a", "b", "c"]
    if request.param == "phoenix":
        return PhoenixOnlyAddLayers(gene_labels, ("test_cond",), hidden_size=2)
    elif request.param == "mlp":
        return MLP(
            gene_labels,
            ("test_cond",),
            hidden_size=2,
            nr_layers=2,
            synthesis_only_positive=True,
        )
    elif request.param == "cellbox":
        return CellBox(gene_labels, ("test_cond",))


@pytest.fixture
def init_values():
    return torch.tensor(
        [
            1.0,
            2.0,
            3.0,
        ]
    ).unsqueeze(0)


def test_integrate_model(ode_model, init_values):
    ode_model.set_conditions(torch.tensor([[0.0] * 4]))
    t = torch.tensor([0.0, 0.5, 1.0, 1.5])
    output = ode_model.integrate(init_values, t)
    assert output.shape == (1, 3, 4)


def test_ko_model(ode_model, init_values):
    # Knockout gene
    ode_model.set_conditions(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    forward_pass_with_ko = ode_model.forward(torch.tensor([0]), init_values)
    assert forward_pass_with_ko[0, 1] < 0
