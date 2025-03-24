import torch
from mlmmint.Models.MachineLearningModel import NeuralODE


def test_create_neuralODE():
    gene_labels = ["a", "b", "c"]
    my_neural_ode = NeuralODE(gene_labels)
    init_values = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([0.0, 0.5, 1.0, 1.5])
    output = my_neural_ode.integrate(init_values, t)
    assert output.shape == (3, 4)
