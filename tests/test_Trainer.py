from copy import deepcopy
from unittest.mock import MagicMock
import numpy as np
import pytest
from pytest import approx
import torch

from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.Models.CellBox import CellBox
from halcyon.Models.NeuralODE import PhoenixOnlyAddLayers, PhoenixAddAndProdLayers
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
    assert my_trainer.calculate_loss_expression_vs_pred(
        expressions_gt, expressions_gt
    ) == approx(0.0)


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


@pytest.mark.parametrize(
    "model_class, model_args",
    [
        (CellBox, {"self_activations": True}),
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
def train_job_with_loaded_cellbox(sim_result) -> TrainingJob:
    train_job = TrainingJob(
        CellBox,
        sim_result,
        nr_training_steps=2,
        optimizer_kwargs={"lr": 0.1},
    )
    train_job.run()
    return train_job


def test_load_double_ko_tensors(train_job_with_loaded_cellbox: TrainingJob):
    double_ko_expressions = train_job_with_loaded_cellbox.get_double_ko_files()
    exp_condition, expression_data = double_ko_expressions[0]
    # Only a 1 at position 0 and 3 because those are the KO genes
    assert torch.allclose(
        exp_condition,
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    expected_expression = torch.tensor(
        [
            [
                7.2333e-01,
                2.5859e-01,
                9.2444e-02,
                3.3048e-02,
                1.1815e-02,
                4.2236e-03,
                1.5099e-03,
                5.3980e-04,
                1.9300e-04,
                6.9000e-05,
                2.4700e-05,
                8.8000e-06,
                3.2000e-06,
                1.1000e-06,
                4.0000e-07,
                1.0000e-07,
                1.0000e-07,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                6.3392e-01,
                6.3358e-01,
                6.3338e-01,
                6.3325e-01,
                6.3317e-01,
                6.3313e-01,
                6.3310e-01,
                6.3308e-01,
                6.3307e-01,
                6.3306e-01,
                6.3306e-01,
                6.3306e-01,
                6.3306e-01,
                6.3306e-01,
                6.3306e-01,
                6.3306e-01,
                6.3306e-01,
                6.3306e-01,
                6.3305e-01,
                6.3305e-01,
                6.3305e-01,
            ],
            [
                9.4803e-01,
                9.7085e-01,
                9.8057e-01,
                9.7110e-01,
                9.1493e-01,
                8.5850e-01,
                8.3716e-01,
                8.3108e-01,
                8.2946e-01,
                8.2903e-01,
                8.2892e-01,
                8.2890e-01,
                8.2889e-01,
                8.2889e-01,
                8.2889e-01,
                8.2889e-01,
                8.2889e-01,
                8.2889e-01,
                8.2889e-01,
                8.2889e-01,
                8.2889e-01,
            ],
            [
                7.4355e-01,
                2.1688e-01,
                6.3262e-02,
                1.8453e-02,
                5.3824e-03,
                1.5700e-03,
                4.5790e-04,
                1.3360e-04,
                3.9000e-05,
                1.1400e-05,
                3.3000e-06,
                1.0000e-06,
                3.0000e-07,
                1.0000e-07,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                1.3590e-01,
                1.3195e-01,
                1.3661e-01,
                1.3955e-01,
                1.3996e-01,
                1.4000e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
                1.4001e-01,
            ],
            [
                9.9317e-01,
                9.9318e-01,
                9.9319e-01,
                9.9319e-01,
                9.9319e-01,
                9.9319e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
                9.9320e-01,
            ],
            [
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
                1.1174e-02,
            ],
            [
                2.3878e-01,
                3.6246e-01,
                6.4116e-01,
                8.5048e-01,
                9.4629e-01,
                9.8198e-01,
                9.9416e-01,
                9.9814e-01,
                9.9941e-01,
                9.9982e-01,
                9.9994e-01,
                9.9998e-01,
                9.9999e-01,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
            ],
            [
                2.9734e-01,
                2.0553e-01,
                8.6743e-02,
                4.5236e-02,
                3.2650e-02,
                2.8864e-02,
                2.7725e-02,
                2.7383e-02,
                2.7280e-02,
                2.7249e-02,
                2.7239e-02,
                2.7237e-02,
                2.7236e-02,
                2.7236e-02,
                2.7236e-02,
                2.7236e-02,
                2.7236e-02,
                2.7236e-02,
                2.7236e-02,
                2.7236e-02,
                2.7236e-02,
            ],
            [
                9.9509e-01,
                9.9508e-01,
                9.9508e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
                9.9507e-01,
            ],
        ]
    )

    assert torch.allclose(expression_data, expected_expression, rtol=1e-03, atol=1e-03)


def test_losses_on_double_ko(train_job_with_loaded_cellbox: TrainingJob):
    losses = train_job_with_loaded_cellbox.losses_on_double_ko()
    assert np.allclose(
        losses,
        np.array(
            [
                0.09203624725341797,
                0.11233041435480118,
                0.07290700823068619,
                0.06423024833202362,
                0.04246918857097626,
                0.1192648783326149,
                0.04892643541097641,
                0.05940413102507591,
                0.0804734006524086,
            ]
        ),
        rtol=1e-02,
        atol=1e-02,
    )
