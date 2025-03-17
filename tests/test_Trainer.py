from copy import deepcopy
from unittest.mock import MagicMock
from pytest import approx
import torch

from mlmmint.Models.MechanisticModel import TorchLinearODE
from mlmmint.Trainer import Trainer
from tests.create_dummy_data import get_synthetic_data


def test_calculate_loss():
    _, _, expressions = get_synthetic_data()
    mock_model = MagicMock()
    mock_model.modelled_gene_labels.return_value = ["G1", "G3"]
    mock_model.parameters.return_value = [torch.randn(2, 2), torch.randn(3, 3)]
    my_trainer = Trainer(model=mock_model, expressions=expressions)  # type: ignore
    expressions_gt = torch.tensor(
        [
            [0.1, 0.11151179, 0.21647672, 0.51517834],
            [0.3, 0.30463339, 0.34882503, 0.43889286],
        ]
    )
    assert my_trainer.calculate_loss(expressions_gt) == approx(0.0)


def test_integrate_model():
    gt_regulatory_network, interactions, expressions = get_synthetic_data()
    correct_model = TorchLinearODE(gt_regulatory_network)
    my_trainer = Trainer(correct_model, expressions)
    init_values, t = my_trainer.get_initial_states_and_time_points()
    output = my_trainer.integrate_model(init_values, t)
    assert torch.allclose(my_trainer.calculate_loss(output), torch.tensor([0.0]))


def test_model_with_more_genes_in_expressions():
    # Get an expression dataset that contains more genes than are in the network
    gt_regulatory_network, interactions, expressions = get_synthetic_data()
    expressions = expressions._add_one_gene_expression("G6", 2)
    correct_model = TorchLinearODE(gt_regulatory_network)
    my_trainer = Trainer(correct_model, expressions)
    init_values, t = my_trainer.get_initial_states_and_time_points()
    output = my_trainer.integrate_model(init_values, t)

    assert torch.allclose(my_trainer.calculate_loss(output), torch.tensor([0.0]))


def test_train_model():
    gt_regulatory_network, interactions, expressions = get_synthetic_data()
    correct_model = TorchLinearODE(gt_regulatory_network)
    # Set all values to zero in weight matrix
    wrong_model = deepcopy(correct_model)
    wrong_model._make_weight_matrix_zeroes()
    assert torch.allclose(
        wrong_model.weight_matrix, torch.zeros_like(wrong_model.weight_matrix)
    )
    my_trainer = Trainer(wrong_model, expressions)
    loss_history = my_trainer.train(200)
    assert loss_history[-1] < 1e-3
    # assert torch.allclose(wrong_model.weight_matrix, correct_model.weight_matrix, atol=0.1)
