from copy import deepcopy
from unittest.mock import MagicMock
import numpy as np
from pytest import approx
import torch

from mlmmint.Models.MachineLearningModel import CellBox, NeuralODE
from mlmmint.Models.MechanisticModel import TorchLinearODE
from mlmmint.Trainer import Trainer
from tests.create_dummy_data import create_synthetic_expressions, get_synthetic_data


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
    output = my_trainer.integrate_model_with_init_and_t_from_expressions()
    assert torch.allclose(my_trainer.calculate_loss(output), torch.tensor([0.0]))


def test_model_with_more_genes_in_expressions():
    # Get an expression dataset that contains more genes than are in the network
    gt_regulatory_network, interactions, expressions = get_synthetic_data()
    expressions = expressions._add_one_gene_expression("G6", 2)
    correct_model = TorchLinearODE(gt_regulatory_network)
    my_trainer = Trainer(correct_model, expressions)
    output = my_trainer.integrate_model_with_init_and_t_from_expressions()

    assert torch.allclose(my_trainer.calculate_loss(output), torch.tensor([0.0]))


def test_train_mechanistic_linear_model():
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


def test_train_neural_ode():
    expressions = create_synthetic_expressions()
    neural_ode_model = NeuralODE(expressions.genes)
    my_trainer = Trainer(neural_ode_model, expressions)
    loss_history = my_trainer.train(5)
    # Assert loss is decreasing
    assert all(x > y for x, y in zip(loss_history, loss_history[1:]))
    # assert torch.allclose(wrong_model.weight_matrix, correct_model.weight_matrix, atol=0.1)


def test_train_cellbox():
    expressions = create_synthetic_expressions()
    cellbox = CellBox(expressions.genes)
    cellbox._set_epsilon_to_zero()
    init_epsilon = deepcopy(cellbox.epsilon)
    init_decay = deepcopy(cellbox.decay_params)
    my_trainer = Trainer(cellbox, expressions)
    loss_history = my_trainer.train(5)
    # Assert loss is decreasing
    assert all(x > y for x, y in zip(loss_history, loss_history[1:]))
    # Check if epsilon and decay rate changes
    assert not torch.allclose(cellbox.epsilon, init_epsilon)
    assert not torch.allclose(cellbox.decay_params, init_decay)


def test_plot_fit_vs_real_data():
    gt_regulatory_network, _, expressions = get_synthetic_data()
    correct_model = TorchLinearODE(gt_regulatory_network)
    my_trainer = Trainer(correct_model, expressions)
    out_fig, out_axs = my_trainer.plot_fit_vs_real()
    assert out_fig is not None
    # Check if times both match
    gt_time = [0.0, 0.11618519612311953, 1.2780371573543148, 5.0]
    for ax in out_axs:
        assert np.allclose(ax.get_lines()[0].get_data()[0], gt_time)
    # Check if y values match for all genes'
    for i, _ in enumerate(expressions.genes):
        assert np.allclose(
            out_axs[0].get_lines()[i].get_data()[1],
            out_axs[1].get_lines()[i].get_data()[1],
            atol=1e-3,
        )


def test_with_regularisation():
    gt_regulatory_network, _, expressions = get_synthetic_data()
    correct_model = TorchLinearODE(gt_regulatory_network)
    my_trainer = Trainer(
        correct_model, expressions, optimizer_kwargs={"weight_decay": 1}
    )
    assert my_trainer.optimizer.state_dict()["param_groups"][0]["weight_decay"] == 1
