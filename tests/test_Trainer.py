from copy import deepcopy
import pytest
from pytest import approx
import torch

from halcyon.ModelRunner import ModelRunner


from torch.testing import assert_close


def test_integration_train_and_load(trainers_for_all_models: list[ModelRunner]):
    for trainer in trainers_for_all_models:
        untrained_trainer = deepcopy(trainer)

        trainer.run_for_parallelisation_and_save()

        is_decreasing = trainer.loss_history[0] > trainer.loss_history[-1]
        assert is_decreasing, f"Loss is not decreasing for {trainer}"
        trained_model_state_dicts = {
            k: v.clone().detach() for k, v in trainer.model.state_dict().items()
        }

        untrained_trainer.load_if_exists()
        loaded_state_dict = untrained_trainer.model.state_dict()
        for weight_key in trained_model_state_dicts:
            if weight_key.startswith("_mm._synthesis_module") or weight_key.startswith(
                "_mm._decay_module"
            ):
                continue
            assert_close(
                trained_model_state_dicts[weight_key], loaded_state_dict[weight_key]
            ), f"Mismatch in {weight_key}"


def test_load_double_ko_tensors(trainer_with_loaded_cellbox: ModelRunner):
    double_ko_expressions = (
        trainer_with_loaded_cellbox.sim_result.get_double_ko_expressiondata()
    )
    exp_condition, expression_data = double_ko_expressions[0]
    # Only a 1 at position 0 and 3 because those are the KO genes
    assert_close(
        exp_condition,
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    # Check if KO genes are indeed 0 at end
    assert_close(
        expression_data[:, -1],
        torch.tensor(
            [
                0.0000,
                0.7354,
                0.0908,
                0.0000,
                0.0152,
                0.9581,
                0.9403,
                0.0124,
                1.0000,
                0.0096,
            ],
        ),
        rtol=1e-03,
        atol=1e-03,
    )


def test_losses_on_double_ko(trainer_with_loaded_cellbox: ModelRunner):
    losses = trainer_with_loaded_cellbox.losses_on_double_ko()
    assert len(losses) == 10
