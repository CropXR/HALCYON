from matplotlib.axes import Axes
import numpy as np
from tests.create_dummy_data import create_synthetic_expressions, get_synthetic_data


def test_add_genes_to_expressions():
    expressions = create_synthetic_expressions()
    expressions = expressions._add_one_gene_expression("G6", 2)
    assert "G6" in expressions.genes
    assert all(expressions.df.loc["G6"] == 2)


def test_extract_info():
    expressions = create_synthetic_expressions()
    initial_states = expressions._get_initial_states()
    time_points = expressions._get_time_points()
    time_points_gt = np.array([0.0, 0.1161852, 1.27803716, 5.0])
    gt_initial_states = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.allclose(time_points, time_points_gt, atol=1e-5)
    assert np.allclose(initial_states, gt_initial_states, atol=1e-5)


def test_filter_by_only_present_genes():
    gene_network, interactions, expressions = get_synthetic_data()
    expressions._add_one_gene_expression("G6", 2)
    # Reindex the dataframe to move the newly added gene to the top.
    # This ensures that the test checks for any potential index errors
    # that might occur due to reordering of the dataframe.
    np.random.seed(174)
    shuffled_genes = np.random.permutation(expressions.genes)
    expressions.df = expressions.df.reindex(shuffled_genes)
    expressions = expressions.match_genes_to_list(gene_network.node_names)
    assert [gene in ["G1", "G2", "G3", "G4", "G5"] for gene in expressions.genes]
    assert "G6" not in expressions.genes
    assert gene_network.node_names == expressions.genes


def test_plot_expressions():
    _, _, expressions = get_synthetic_data()
    plot = expressions.plot_expressions()
    assert isinstance(plot, Axes)


def test_get_long_form_df():
    _, _, expressions = get_synthetic_data()
    long_df = expressions._return_long_df()
    assert list(long_df.columns) == [
        expressions.gene_name,
        expressions.time_name,
        expressions.expression_name,
    ]
    assert long_df.shape == (len(expressions.df.index) * len(expressions.df.columns), 3)
