from tests.create_dummy_data import create_synthetic_expressions


def test_add_genes_to_expressions():
    expressions = create_synthetic_expressions()
    expressions = expressions._add_one_gene_expression("G6", 2)
    assert "G6" in expressions.genes
    assert all(expressions.df.loc["G6"] == 2)
