from halcyon.Models.BaseLineModels import AlwaysDerivativeZero
from torch.testing import assert_close
import torch


def test_return_derivative_zero_model():
    model = AlwaysDerivativeZero(
        modelled_gene_labels=["a", "b", "c"], condition_names=[]
    )
    t = torch.tensor([0.0])
    state = torch.tensor([[1.0] * 3])
    derivative = model(t, state)
    assert_close(derivative, torch.tensor([[0.0] * 3]))
