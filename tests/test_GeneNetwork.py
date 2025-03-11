import pytest
from tests.create_dummy_data import create_synthetic_gene_network


def test_get_ingoing_interactions():
    my_network, _ = create_synthetic_gene_network()
    ingoing_interactions = my_network.get_ingoing_interactions("G1")
    assert ingoing_interactions[0] == ("G5", "G1", {"influence": 0.2})


def test_add_gene_to_network():
    my_network, _ = create_synthetic_gene_network()
    my_network.add_edge(
        ("G5", "G6"), attributes={"influence": 0.1}, contains_new_genes=True
    )
    assert "G6" in my_network.genes


def test_adding_unknown_gene_raises_error():
    my_network, _ = create_synthetic_gene_network()
    with pytest.raises(AssertionError):
        my_network.add_edge(
            ("G5", "G6"), attributes={"influence": 0.1}, contains_new_genes=False
        )
