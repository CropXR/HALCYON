import numpy as np
import torch

from torch.testing import assert_close


def test_extract_info(create_synthetic_expressions):
    expressions = create_synthetic_expressions
    # First sample, expression data, all genes, timepoint 0
    initial_states = expressions[0][1][:, 0]
    time_points = expressions._get_time_points()
    time_points_gt = np.array([0.0, 0.1161852, 1.27803716, 5.0])
    gt_initial_states = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.allclose(time_points, time_points_gt, atol=1e-5)
    assert np.allclose(initial_states, gt_initial_states, atol=1e-5)


def test_match_genes_to_list(get_synthetic_data):
    _, _, expressions = get_synthetic_data
    expressions.match_genes_to_list(["G5", "G4", "G2", "G1", "G3"])
    assert expressions.genes == ["G5", "G4", "G2", "G1", "G3"]
    assert_close(expressions[0][1][:, 0], torch.tensor([0.5, 0.4, 0.2, 0.1, 0.3]))
