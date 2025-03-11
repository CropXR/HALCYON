import numpy as np
from mlmmint.Models.MechanisticModel import LinearMechanisticModel
from tests.create_dummy_data import (
    create_synthetic_gene_network,
    create_synthetic_expressions,
)


def test_make_mechanistic_model_from_gene_network():
    gene_network, interactions = create_synthetic_gene_network()
    expressions = create_synthetic_expressions()
    my_model = LinearMechanisticModel(gene_network, expressions)

    time_points_gt = [0.0, 0.1161852, 1.27803716, 5.0]
    gt_initial_states = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    output = my_model.integrate(
        initial_state=gt_initial_states, timespan=[0, 5], t_eval=time_points_gt
    )

    expressions_gt = np.array(
        [
            [0.1, 0.11151179, 0.21647672, 0.51517834],
            [0.2, 0.19877107, 0.17955425, 0.0442817],
            [0.3, 0.30463339, 0.34882503, 0.43889286],
            [0.4, 0.38595, 0.2339324, -0.36489983],
            [0.5, 0.49086801, 0.41844875, 0.4587897],
        ]
    )
    assert np.allclose(output.t, time_points_gt, atol=1e-5)
    assert np.allclose(output.y, expressions_gt, atol=1e-5)


def test_filter_by_only_present_genes():
    gene_network, interactions = create_synthetic_gene_network()
    expressions = create_synthetic_expressions()
    expressions._add_one_gene_expression("G6", 2)
    # Reindex the dataframe to move the newly added gene to the top.
    # This ensures that the test checks for any potential index errors
    # that might occur due to reordering of the dataframe.
    np.random.seed(174)
    shuffled_genes = np.random.permutation(expressions.genes)
    expressions.df = expressions.df.reindex(shuffled_genes)
    expressions = expressions.keep_genes_in_list(gene_network.genes)
    assert all([gene in ["G1", "G2", "G3", "G4", "G5"] for gene in expressions.genes])
