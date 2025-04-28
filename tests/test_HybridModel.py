from unittest.mock import MagicMock

import torch
from mlmmint.Models.HybridModel import HybridModel
from mlmmint.Models.MachineLearningModel import NeuralODE
from mlmmint.Models.MechanisticModel import TorchLinearODE
from mlmmint.Trainer import Trainer
from tests.create_dummy_data import get_synthetic_data


def test_merge_modelled_genes():
    mm_model = MagicMock()
    mm_model.modelled_gene_labels = ["a", "b", "c"]
    ml_model = MagicMock()
    ml_model.modelled_gene_labels = ["c", "d"]
    hybrid_model = HybridModel(mm_model, ml_model)
    assert set(hybrid_model.modelled_gene_labels) == {"a", "b", "c", "d"}


def test_hybrid_model_gene_split():
    gt_regulatory_network, interactions, expression_dataset = get_synthetic_data()
    subset_regulatory_network = gt_regulatory_network.select_subset(["G1", "G4", "G5"])
    env_conditions, expressions = expression_dataset[0]

    # imitate batch
    env_conditions, expressions = env_conditions.unsqueeze(0), expressions.unsqueeze(0)

    ml_model = NeuralODE(
        expression_dataset.genes, expression_dataset.conditions[0].all_environments
    )
    mm_model = TorchLinearODE(subset_regulatory_network)
    hybrid_model = HybridModel(mm_model, ml_model)
    # Ensure that they handle the different order of genes well
    assert torch.allclose(hybrid_model.mm_indices, torch.tensor([0, 3, 4]))
    assert torch.allclose(hybrid_model.ml_indices, torch.tensor([0, 1, 2, 3, 4]))
    assert [
        hybrid_model.modelled_gene_labels[i] for i in hybrid_model.mm_indices.tolist()
    ] == ["G1", "G4", "G5"]
    hybrid_model.set_conditions(env_conditions)
    assert torch.allclose(
        hybrid_model._get_mm_forward(0, expressions[:, :, 0]),
        torch.tensor([0.1000, 0.0000, -0.0800]),
    )

    one_forward = hybrid_model.forward(t=torch.tensor([0]), state=expressions[:, :, 0])
    assert one_forward.shape == (1, 5)

    trainer = Trainer(hybrid_model, expression_dataset)

    integration_output = trainer.integrate_model(expressions[:, :, 0])
    # Check if at least shape and initial gene expressions match
    assert tuple(integration_output.shape) == (1, 5, 4)
    assert torch.allclose(
        integration_output[0, :, 0],
        torch.tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000]),
    )
