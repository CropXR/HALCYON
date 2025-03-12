import numpy as np
from tests.create_dummy_data import (
    get_synthetic_data,
)


def test_filter_by_only_present_genes():
    gene_network, interactions, expressions = get_synthetic_data()
    expressions._add_one_gene_expression("G6", 2)
    # Reindex the dataframe to move the newly added gene to the top.
    # This ensures that the test checks for any potential index errors
    # that might occur due to reordering of the dataframe.
    np.random.seed(174)
    shuffled_genes = np.random.permutation(expressions.genes)
    expressions.df = expressions.df.reindex(shuffled_genes)
    expressions = expressions.keep_genes_in_list(gene_network.node_names)
    assert ["G1", "G2", "G3", "G4", "G5"] == expressions.genes
