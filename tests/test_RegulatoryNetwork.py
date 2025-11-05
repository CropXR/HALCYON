import pytest


def test_get_ingoing_interactions(create_synthetic_gene_network):
    my_network, _ = create_synthetic_gene_network
    ingoing_interactions = my_network.get_ingoing_interactions("G1")
    assert ingoing_interactions[0] == ("G5", "G1", {"influence": 0.2})


def test_add_gene_to_network(create_synthetic_gene_network):
    my_network, _ = create_synthetic_gene_network
    my_network.add_edge(
        ("G5", "G6"), attributes={"influence": 0.1}, contains_new_genes=True
    )
    assert "G6" in my_network.node_names


def test_adding_unknown_gene_raises_error(create_synthetic_gene_network):
    my_network, _ = create_synthetic_gene_network
    with pytest.raises(AssertionError):
        my_network.add_edge(
            ("G5", "G6"), attributes={"influence": 0.1}, contains_new_genes=False
        )


def test_get_subset_of_network(create_synthetic_gene_network):
    my_network, _ = create_synthetic_gene_network
    selected_genes = ["G1", "G2", "G3"]
    subset = my_network.select_subset(selected_genes)
    assert len(subset.node_names) == 3
