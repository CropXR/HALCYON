from copy import deepcopy
from unittest.mock import MagicMock
import numpy as np
import pytest
from pytest import approx
import torch

from halcyon.Models.CellBox import CellBox
from halcyon.Models.NeuralODE import PhoenixOnlyAddLayers, PhoenixAddAndProdLayers
from halcyon.Models.MechanisticModel import TorchLinearODE
from halcyon.Simulators.GNWSimRunner import SimulationResult
from halcyon.Trainer import Trainer, TrainingJob
from tests.create_dummy_data import (
    create_synthetic_expressions,
    get_synthetic_data,
)
from tests.create_dummy_data import expression_data_from_gnw
from tests.test_GNWSimRunner import sim_result


@pytest.fixture
def cellbox_from_expression():
    expressions = create_synthetic_expressions()
    return CellBox(expressions.genes, ("test", "test2"))


def test_calculate_loss():
    _, _, expressions = get_synthetic_data()
    mock_model = MagicMock()
    mock_model.modelled_gene_labels.return_value = ["G1", "G3"]
    mock_model.parameters.return_value = [torch.randn(2, 2), torch.randn(3, 3)]
    my_trainer = Trainer(model=mock_model, dataset=expressions)  # type: ignore
    expressions_gt = torch.tensor(
        [
            [0.1, 0.11151179, 0.21647672, 0.51517834],
            [0.3, 0.30463339, 0.34882503, 0.43889286],
        ]
    )
    assert my_trainer.calculate_loss(expressions_gt, expressions_gt) == approx(0.0)


def test_integrate_model():
    gt_regulatory_network, interactions, data = get_synthetic_data()
    correct_model = TorchLinearODE(gt_regulatory_network)
    my_trainer = Trainer(correct_model, data)
    gt_expressions = data[0][1]
    init_conditions = gt_expressions[:, 0].unsqueeze(0)
    output = my_trainer.integrate_model(init_conditions)
    assert torch.allclose(
        my_trainer.calculate_loss(
            output, gt_expressions.unsqueeze(0).to(correct_model.device)
        ),
        torch.tensor([0.0], device=correct_model.device),
    )


def test_train_neural_ode():
    expressions = create_synthetic_expressions()
    neural_ode_model = PhoenixOnlyAddLayers(
        expressions.genes, expressions.all_environments, hidden_size=2
    )
    my_trainer = Trainer(neural_ode_model, expressions, batch_size=1)
    init_weights = torch.cat(
        [i.clone().detach().flatten() for i in neural_ode_model.neural_net.parameters()]
    )
    init_decay = neural_ode_model.decay_params.clone().detach()
    my_trainer.train(5)
    # Assert loss is decreasing
    assert all(
        x > y for x, y in zip(my_trainer.loss_history, my_trainer.loss_history[1:])
    )
    assert not torch.allclose(
        init_weights,
        torch.cat(
            [i.detach().flatten() for i in neural_ode_model.neural_net.parameters()]
        ),
    )
    assert not torch.allclose(init_decay, neural_ode_model.decay_params.detach())


def test_train_cellbox(cellbox_from_expression):

    cellbox_from_expression._set_weight_matrix_to_ones()
    cellbox_from_expression._set_epsilon_to_zero()
    init_epsilon = deepcopy(cellbox_from_expression.epsilon)
    init_decay = deepcopy(cellbox_from_expression.decay_params)
    my_trainer = Trainer(cellbox_from_expression, create_synthetic_expressions())
    my_trainer.train(5)
    # Assert loss is decreasing
    assert all(
        x > y for x, y in zip(my_trainer.loss_history, my_trainer.loss_history[1:])
    )
    # Check if epsilon and decay rate changes
    assert not torch.allclose(cellbox_from_expression.epsilon, init_epsilon)
    assert not torch.allclose(cellbox_from_expression.decay_params, init_decay)


def test_train_cellbox_ko_data(expression_data_from_gnw):
    cellbox = CellBox(
        expression_data_from_gnw.genes, expression_data_from_gnw.all_environments
    )
    # cellbox_from_expression._set_epsilon_to_zero()
    cellbox._set_weight_matrix_to_ones()
    init_epsilon = deepcopy(cellbox.epsilon)
    init_decay = deepcopy(cellbox.decay_params)
    init_w = deepcopy(cellbox.W)
    my_trainer = Trainer(cellbox, expression_data_from_gnw, batch_size=2)
    my_trainer.train(3)
    # Assert loss is decreasing
    assert all(
        x > y for x, y in zip(my_trainer.loss_history, my_trainer.loss_history[1:])
    )
    # Check if epsilon and decay rate changes
    assert not torch.allclose(cellbox.epsilon, init_epsilon)
    assert not torch.allclose(cellbox.decay_params, init_decay)
    assert not torch.allclose(cellbox.W, init_w)


def test_plot_fit_vs_real_data(cellbox_from_expression):
    gt_regulatory_network, _, expressions = get_synthetic_data()
    my_trainer = Trainer(cellbox_from_expression, expressions)
    out_fig, out_axs = my_trainer.plot_fit_vs_real()
    assert out_fig is not None


def test_with_regularisation():
    gt_regulatory_network, _, expressions = get_synthetic_data()
    correct_model = TorchLinearODE(gt_regulatory_network)
    my_trainer = Trainer(
        correct_model, expressions, optimizer_kwargs={"weight_decay": 1}
    )
    assert my_trainer.optimizer.state_dict()["param_groups"][0]["weight_decay"] == 1


@pytest.mark.parametrize(
    "model_class, model_args",
    [
        (CellBox, None),
        (PhoenixOnlyAddLayers, {"hidden_size": 2}),
        (PhoenixAddAndProdLayers, {"hidden_size": 2}),
    ],
)
def test_train_job_and_load_model(model_class, model_args, sim_result):
    train_job = TrainingJob(
        model_class,
        sim_result,
        nr_training_steps=2,
        optimizer_kwargs={"lr": 0.1},
        model_kwargs=model_args,
    )
    train_job.run()
    og_model_state_dicts = {
        k: v.clone().detach() for k, v in train_job.trainer.model.state_dict().items()
    }
    train_job = TrainingJob(
        model_class,
        sim_result,
        nr_training_steps=1,
        optimizer_kwargs={"lr": 0.1},
        model_kwargs=model_args,
    )
    train_job.load_if_exists()
    loaded_state_dict = train_job.trainer.model.state_dict()
    for weight_key in og_model_state_dicts:
        assert torch.allclose(
            og_model_state_dicts[weight_key], loaded_state_dict[weight_key]
        ), f"Mismatch in {weight_key}"


@pytest.fixture
def train_job_with_loaded_cellbox(sim_result):
    train_job = TrainingJob(
        CellBox,
        sim_result,
        nr_training_steps=2,
        optimizer_kwargs={"lr": 0.1},
    )
    train_job.run()
    return train_job


def test_verify_on_double_ko(train_job_with_loaded_cellbox):
    trainer = train_job_with_loaded_cellbox.trainer
    trainer.verify_on_double_ko()
