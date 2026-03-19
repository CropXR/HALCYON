from torch import tensor
from torch.testing import assert_close
from halcyon.Models.Losses import MSELossFlexible, MSELossPlusL1
import pytest


@pytest.fixture()
def mse_loss():
    return MSELossFlexible()


@pytest.fixture()
def true_values():
    return tensor([1.0, 2.0, 3.0])


@pytest.fixture()
def wrong_values():
    return tensor([2.0, 2.0, 3.0])


def test_l1_loss(true_values, wrong_values, ode_model, mse_loss):
    l1_lambda_0 = MSELossPlusL1(lambda_param=0)
    normal_mse = mse_loss(true_values, wrong_values, ode_model)
    l1_loss_with_lambda_0 = l1_lambda_0(true_values, wrong_values, ode_model)
    assert_close(l1_loss_with_lambda_0, normal_mse)
    l1_loss_with_lambda_0.backward()

    l1_lambda_1 = MSELossPlusL1(lambda_param=1)
    l1_loss_with_lambda_1 = l1_lambda_1(true_values, wrong_values, ode_model)
    assert l1_loss_with_lambda_0 > l1_loss_with_lambda_1
    l1_loss_with_lambda_1.backward()


def test_mse_perfect_prediction(true_values, mse_loss):
    assert_close(mse_loss(true_values, true_values), tensor(0.0))
