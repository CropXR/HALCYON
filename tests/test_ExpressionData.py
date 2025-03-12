import numpy as np
from tests.create_dummy_data import create_synthetic_expressions


def test_add_genes_to_expressions():
    expressions = create_synthetic_expressions()
    expressions = expressions._add_one_gene_expression("G6", 2)
    assert "G6" in expressions.genes
    assert all(expressions.df.loc["G6"] == 2)


def test_extract_info():
    expressions = create_synthetic_expressions()
    initial_states = expressions.get_initial_states()
    time_points = expressions.get_time_points()
    time_points_gt = np.array([0.0, 0.1161852, 1.27803716, 5.0])
    gt_initial_states = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.allclose(time_points, time_points_gt, atol=1e-5)
    assert np.allclose(initial_states, gt_initial_states, atol=1e-5)
