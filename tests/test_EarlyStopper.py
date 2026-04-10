import pytest

from halcyon.ModelRunner import EarlyStopper


@pytest.fixture
def early_stopper() -> EarlyStopper:
    return EarlyStopper(patience=3, min_delta=1e-4)


def test_does_not_stop_while_improving(early_stopper: EarlyStopper):
    losses = [1.0, 0.9, 0.8, 0.7]
    assert not any(early_stopper.check_if_should_stop(loss) for loss in losses)


def test_stops_after_patience_exceeded(early_stopper: EarlyStopper):
    early_stopper.check_if_should_stop(1.0)
    early_stopper.check_if_should_stop(1.0)
    early_stopper.check_if_should_stop(1.0)
    assert early_stopper.check_if_should_stop(1.0)
