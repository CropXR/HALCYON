from unittest.mock import MagicMock

import numpy as np
from halcyon.Evaluator import Evaluator
from halcyon.Models.CellBox import CellBox
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from tests.test_fixtures import get_synthetic_data
import networkx as nx


def test_compare_inferred_network():
    gt_regulatory_network, interactions, expressions = get_synthetic_data()

    my_evaluator = Evaluator(gt_regulatory_network, expressions)
    mock_model = MagicMock()
    mock_model.get_inferred_network.return_value = gt_regulatory_network
    # Edge jaccard score
    comparison_output = my_evaluator.compare_edge_jaccard_gt_vs_inferred_network(
        mock_model
    )
    assert comparison_output == 1

    # Empty network
    mock_model.get_inferred_network.return_value = RegulatoryNetwork(nx.DiGraph())
    comparison_output = my_evaluator.compare_edge_jaccard_gt_vs_inferred_network(
        mock_model
    )
    assert comparison_output == 0


def test_roc_auc_inferred_network():
    gt_regulatory_network, interactions, expressions = get_synthetic_data()
    my_evaluator = Evaluator(gt_regulatory_network, expressions)
    cellbox = CellBox(gt_regulatory_network.node_names, [], self_activations=False)
    cellbox._set_weight_matrix_to_ones()
    comparison_output = my_evaluator.auroc_score_edge_detection(cellbox)
    assert np.allclose(comparison_output, 0.625)
