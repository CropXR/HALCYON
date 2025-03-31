import torch
from mlmmint.Models.MachineLearningModel import CellBox, NeuralODE


def test_create_neuralODE():
    gene_labels = ["a", "b", "c"]
    my_neural_ode = NeuralODE(gene_labels)
    init_values = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([0.0, 0.5, 1.0, 1.5])
    output = my_neural_ode.integrate(init_values, t)
    assert output.shape == (3, 4)


def test_cellbox_forward():
    # Test nonlinear envelope
    gene_labels = ["a", "b", "c"]
    my_cellbox = CellBox(gene_labels)
    init_values = torch.tensor([1.0, 2.0, 3.0])
    my_cellbox._set_weight_matrix_to_ones()
    # Time not important
    assert torch.allclose(
        my_cellbox.forward(t=torch.tensor([0]), state=init_values),
        torch.tanh(torch.tensor([5, 4, 3]))
        - torch.nn.functional.softplus(my_cellbox.decay_params, beta=10) * init_values,
    )


def test_cellbox_infer_edges():
    gene_labels = ["a", "b", "c"]
    my_cellbox = CellBox(gene_labels)
    my_cellbox._set_weight_matrix_to_ones()
    my_cellbox.knockout_interaction("a", "c")
    out_network = my_cellbox.get_inferred_network()
    assert set(out_network.get_all_interactions(with_data=False)) == {
        ("b", "c"),
        ("b", "a"),
        ("c", "a"),
        ("a", "b"),
        ("c", "b"),
    }


def test_cellbox_decay():
    gene_labels = ["a", "b", "c"]
    my_cellbox = CellBox(gene_labels)
    init_values = torch.tensor([1.0, 2.0, 3.0])
    my_cellbox._set_decay_matrix_to_ones()
    # Time not important in forward pass
    # Due to softplus of the decay parameters the 'output' of the decay is slightly different
    assert torch.allclose(
        my_cellbox.forward(t=torch.tensor([0]), state=init_values),
        torch.tensor([-1, -2, -3], dtype=torch.float32),
        atol=1e-3,
    )


def test_cellbox_gene_knockout():
    # Test nonlinear envelope
    gene_labels = ["a", "b", "c"]
    my_cellbox = CellBox(gene_labels)
    init_values = torch.tensor([1.0, 2.0, 3.0])
    my_cellbox._set_weight_matrix_to_ones()
    assert my_cellbox.get_w_masked()[0, 1] == 1
    my_cellbox.knockout_interaction("b", "a")
    my_cellbox.knockout_interaction("a", "b")
    assert my_cellbox.get_w_masked()[0, 1] == 0
    assert my_cellbox.get_w_masked()[1, 0] == 0
    # Time not important
    assert torch.allclose(
        my_cellbox.forward(t=torch.tensor([0]), state=init_values),
        torch.tanh(torch.tensor([3, 3, 3]))
        - torch.nn.functional.softplus(my_cellbox.decay_params, beta=10) * init_values,
    )
