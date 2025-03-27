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
