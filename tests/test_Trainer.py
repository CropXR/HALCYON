import numpy as np
from pytest import approx
import torch

from mlmmint.Models.MechanisticModel import TorchLinearODE
from mlmmint.Trainer import Trainer
from tests.create_dummy_data import create_synthetic_expressions, get_synthetic_data


def test_integrate_model():
    gt_regulatory_network, interactions, expressions = get_synthetic_data()
    # expressions = expressions._add_one_gene_expression("obscure", 0.1)
    correct_model = TorchLinearODE(gt_regulatory_network)
    my_trainer = Trainer(correct_model)
    initial_values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    time_points = torch.tensor([0.0, 0.1161852, 1.27803716, 5.0])

    output = my_trainer.integrate_model(correct_model, initial_values, time_points)
    expressions_gt = torch.tensor(
        [
            [0.1, 0.11151179, 0.21647672, 0.51517834],
            [0.2, 0.19877107, 0.17955425, 0.0442817],
            [0.3, 0.30463339, 0.34882503, 0.43889286],
            [0.4, 0.38595, 0.2339324, -0.36489983],
            [0.5, 0.49086801, 0.41844875, 0.4587897],
        ]
    )
    assert torch.allclose(output, expressions_gt, atol=1e-2)


def test_calculate_loss():
    expressions = create_synthetic_expressions()
    my_trainer = Trainer(None)  # type: ignore
    expressions_gt = torch.tensor(
        [
            [0.1, 0.11151179, 0.21647672, 0.51517834],
            [0.2, 0.19877107, 0.17955425, 0.0442817],
            [0.3, 0.30463339, 0.34882503, 0.43889286],
            [0.4, 0.38595, 0.2339324, -0.36489983],
            [0.5, 0.49086801, 0.41844875, 0.4587897],
        ]
    )

    assert my_trainer.calculate_loss(expressions, expressions_gt) == approx(0.0)
