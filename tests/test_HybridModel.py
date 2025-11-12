from copy import deepcopy
from unittest.mock import MagicMock

import pytest
import torch
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.Losses import MSELossFlexible, MSELossPlusMachineLearningL2
from halcyon.Models.HybridModel import HybridModel
from halcyon.Models.PhoenixModels import PhoenixAddAndProdLayers, PhoenixOnlyAddLayers
from halcyon.Models.MechanisticModel import PyTorchODE
from halcyon.Trainer import Trainer
from torch.testing import assert_close


def test_merge_modelled_genes():
    mm_model = MagicMock()
    mm_model.modelled_gene_labels = ["a", "b", "c"]
    mm_model.condition_names = ["cond_1", "cond_2"]
    ml_model = MagicMock()
    ml_model.modelled_gene_labels = ["c", "d"]
    ml_model.condition_names = ["cond_1", "cond_2"]
    hybrid_model = HybridModel(mm_model, ml_model)
    assert set(hybrid_model.modelled_gene_labels) == {"a", "b", "c", "d"}


def test_hybrid_does_perfect_fit_if_ml_weights_zero(
    sim_result, test_batch, hybrid_model_with_zero_ml_influence
):
    # Set all per_gene_multipliers to 0 and check if MM model matches data perfectly
    my_trainer = Trainer(hybrid_model_with_zero_ml_influence, sim_result, batch_size=2)
    experiment_data, expression_data = my_trainer.preprocess_batch(test_batch)
    output = my_trainer._calculate_model_prediction_at_conditions(
        experiment_data, expression_data[:, :, 0]
    )
    assert_close(expression_data, output, atol=1e-3, rtol=1e-3)
    loss = my_trainer.calculate_loss_expression_vs_pred(expression_data, output)
    assert_close(loss, torch.tensor(0.0, device=my_trainer.model.device))


def test_most_genes_perfect_fit_if_only_remove_gene_9(
    sim_result,
    hybrid_model_mm_no_node_9,
    test_batch,
):
    hybrid_model_mm_no_node_9 = deepcopy(hybrid_model_mm_no_node_9)
    hybrid_model_mm_no_node_9._ml._set_per_gene_multipliers_to_zero()
    my_trainer = Trainer(
        hybrid_model_mm_no_node_9,
        sim_result,
        batch_size=2,
    )
    experiment_data, expression_data = my_trainer.preprocess_batch(test_batch)
    output = my_trainer._calculate_model_prediction_at_conditions(
        experiment_data, expression_data[:, :, 0]
    )
    # First 9 genes will match perfectly
    assert_close(expression_data[:, :9, :], output[:, :9, :], atol=1e-3, rtol=1e-3)
    loss = my_trainer.calculate_loss_expression_vs_pred(expression_data, output)
    assert_close(
        loss, torch.tensor(0.0382, device=my_trainer.model.device), atol=1e-3, rtol=1e-3
    )


def test_l2_loss(sim_result, hybrid_model_mm_no_node_9, test_batch):
    my_l2_loss = MSELossPlusMachineLearningL2(lambda_param=1)

    my_trainer = Trainer(
        hybrid_model_mm_no_node_9,
        sim_result,
        batch_size=2,
        loss=my_l2_loss,
    )

    loss = my_trainer._calculate_loss_from_batch(test_batch)

    ml_params = [
        p
        for n, p in hybrid_model_mm_no_node_9._ml.named_parameters()
        if not n == "per_gene_multipliers"
    ]
    ml_params_squared = sum((p**2).sum() for p in ml_params)
    total_params = sum(p.numel() for p in ml_params)
    loss_l2 = ml_params_squared / total_params

    inferred_mse = loss - loss_l2

    assert_close(
        loss,
        torch.tensor(inferred_mse + 1 * loss_l2, device=my_trainer.model.device),
        atol=1e-4,
        rtol=1e-4,
    )
    # Changing lambda should change overall loss as such

    my_trainer.loss_fn = MSELossPlusMachineLearningL2(lambda_param=0.5)
    loss = my_trainer._calculate_loss_from_batch(test_batch)
    assert_close(
        loss,
        torch.tensor(inferred_mse + 0.5 * loss_l2, device=my_trainer.model.device),
        atol=1e-4,
        rtol=1e-4,
    )

    my_trainer.loss_fn = MSELossFlexible()
    loss = my_trainer._calculate_loss_from_batch(test_batch)
    assert_close(
        loss,
        torch.tensor(inferred_mse, device=my_trainer.model.device),
        atol=1e-4,
        rtol=1e-4,
    )


def test_hybrid_model_gene_split(hybrid_model_mm_no_node_9):

    assert_close(
        hybrid_model_mm_no_node_9._mm_indices,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        check_device=False,
    )
    assert_close(
        hybrid_model_mm_no_node_9._ml_indices,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        check_device=False,
    )
    assert [
        hybrid_model_mm_no_node_9.modelled_gene_labels[i]
        for i in hybrid_model_mm_no_node_9._mm_indices.tolist()
    ] == [f"Node_{i}" for i in range(9)]
