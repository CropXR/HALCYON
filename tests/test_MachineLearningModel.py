import torch
from halcyon.Models.CellBox import CellBox
from halcyon.Models.NeuralODE import PhoenixOnlyAddLayers
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


@pytest.fixture
def my_cellbox():
    gene_labels = ["a", "b", "c"]
    return CellBox(gene_labels, ["test1", "test2"], self_activations=False)


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


def test_cellbox_forward(my_cellbox):
    init_expr = torch.tensor([1.0, 2.0, 3.0], device=my_cellbox.device).unsqueeze(
        0
    )  # Imitate being one batch
    env_conditions = torch.tensor([0, 0, 0, 0, 0], device=my_cellbox.device).unsqueeze(
        0
    )  # Imitate being one batch
    my_cellbox._set_weight_matrix_to_ones()
    # Time not important
    my_cellbox.set_conditions(env_conditions)
    forward_pass = my_cellbox.forward(t=torch.tensor([0]), state=init_expr)
    assert_close(
        forward_pass,
        torch.tanh(torch.tensor([5.0, 4.0, 3.0], device=my_cellbox.device).unsqueeze(0))
        - torch.nn.functional.softplus(my_cellbox.decay_params, beta=10)
        * torch.tensor([1.0, 2.0, 3.0], device=my_cellbox.device).unsqueeze(0),
    )


def test_cellbox_integrate(my_cellbox):
    init_values = torch.tensor([1.0, 2.0, 3.0], device=my_cellbox.device).unsqueeze(0)
    env_values = torch.tensor([0, 0, 0, 0, 0], device=my_cellbox.device).unsqueeze(0)
    my_cellbox._set_weight_matrix_to_ones()
    my_cellbox._set_decay_matrix_to_zeros()
    my_cellbox.set_conditions(env_values)
    output = my_cellbox.integrate(
        init_values, t_eval=torch.tensor([0.0, 1.0, 2.0], device=my_cellbox.device)
    )
    assert_close(
        output,
        torch.tensor(
            [
                [1.0000, 1.8991, 2.7381],
                [2.0000, 2.8320, 3.6085],
                [3.0000, 3.7639, 4.4780],
            ],
            device=my_cellbox.device,
        ).unsqueeze(0),
        atol=1e-4,
        rtol=1e-4,
    )


def test_cellbox_infer_edges(my_cellbox):
    my_cellbox._set_weight_matrix_to_ones()
    out_network = my_cellbox.get_inferred_network(threshold=0.1)
    assert set(out_network.get_all_interactions(with_data=False)) == {
        ("b", "c"),
        ("b", "a"),
        ("c", "a"),
        ("a", "b"),
        ("a", "c"),
        ("c", "b"),
    }
    with torch.no_grad():
        my_cellbox.W.data[2] = 0  # Zero out third row
    out_network = my_cellbox.get_inferred_network(threshold=0.1)
    assert set(out_network.get_all_interactions(with_data=False)) == {
        ("b", "a"),
        ("c", "a"),
        ("a", "b"),
        ("c", "b"),
    }


def test_cellbox_gene_knockout(my_cellbox):
    # Test nonlinear envelope
    init_values = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device=my_cellbox.device
    )
    env_values = torch.tensor(
        [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]], device=my_cellbox.device
    )
    my_cellbox._set_weight_matrix_to_ones()
    my_cellbox.set_conditions(env_values)
    # Time not important
    forward_pass = my_cellbox.forward(t=torch.tensor([0]), state=init_values)
    assert_close(
        forward_pass[0, :],
        torch.tanh(torch.tensor([5.0, 4.0, 3.0], device=my_cellbox.device))
        - torch.nn.functional.softplus(my_cellbox.decay_params, beta=10)
        * torch.tensor([1.0, 2.0, 3.0], device=my_cellbox.device),
    )

    assert_close(
        forward_pass[1, :],
        torch.tanh(torch.tensor([5.0, 0.0, 3.0], device=my_cellbox.device))
        - torch.nn.functional.softplus(my_cellbox.decay_params, beta=10)
        * torch.tensor([1.0, 2.0, 3.0], device=my_cellbox.device),
    )


def test_cellbox_random_weight_matrix(my_cellbox):
    my_cellbox._set_weight_matrix_to_random()
    assert not torch.allclose(torch.zeros_like(my_cellbox.W), my_cellbox.W, atol=0.1)
