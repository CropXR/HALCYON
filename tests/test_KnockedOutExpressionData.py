from pathlib import Path

import numpy as np
import torch
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions


def test_parse_from_genenetweaver_output():
    input_path = Path(
        "tests/grnsimdummydata/InSilicoSize10-Yeast1_knockout_timeseries.tsv"
    )
    ko_expression_data = (
        ExpressionDataWithConditions.from_genenetweaver_ko_timeseries_output(input_path)
    )
    # For KO of gene 1
    environment, my_expression_data = ko_expression_data[0]
    # Match gene 1 expressions for knocked out G1
    assert torch.allclose(
        environment,
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        atol=1e-4,
    )
    assert torch.allclose(
        my_expression_data[0],
        torch.tensor(
            [
                0.2031,
                0.0027,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
        ),
        atol=1e-4,
    )

    environment, my_expression_data = ko_expression_data[9]
    assert torch.allclose(
        environment,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        atol=1e-4,
    )
    assert torch.allclose(
        my_expression_data[9],
        torch.tensor(
            [
                9.7069e-01,
                3.1041e-01,
                1.0130e-01,
                3.7906e-02,
                1.3047e-02,
                3.6927e-03,
                1.4805e-03,
                2.8280e-04,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ]
        ),
        atol=1e-4,
    )

    goal_gene_list = ["G1", "G10", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]
    ko_expression_data.match_genes_to_list(goal_gene_list)
    for i in range(len(ko_expression_data)):
        environment, my_expression_data = ko_expression_data[i]
        # Index of the knocked out gene
        ko_gene_index = (environment == 1).nonzero().squeeze()
        assert torch.allclose(
            my_expression_data[ko_gene_index][-1], torch.tensor([0.0]), atol=1e-4
        )
