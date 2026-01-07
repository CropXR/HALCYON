import numpy as np
import torch

from torch.testing import assert_close
from numpy.testing import assert_allclose


def test_extract_info(expression_data_from_gnw):
    # First sample, expression data, all genes, timepoint 0

    env_tensor, expression = expression_data_from_gnw[0]
    assert_close(
        env_tensor, torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), check_dtype=False
    )

    initial_states = expression[:, 0]
    assert_close(
        initial_states,
        torch.tensor(
            [
                0.1093,
                0.6281,
                0.7730,
                0.9152,
                0.5725,
                0.6522,
                0.3622,
                0.0797,
                0.7164,
                0.4301,
            ]
        ),
        atol=1e-3,
        rtol=1e-3,
    )

    time_points = expression_data_from_gnw.get_time_points()
    time_points_expected = np.array(
        [
            0.0,
            50.0,
            100.0,
            150.0,
            200.0,
            250.0,
            300.0,
            350.0,
            400.0,
            450.0,
            500.0,
            550.0,
            600.0,
            650.0,
            700.0,
            750.0,
            800.0,
            850.0,
            900.0,
            950.0,
            1000.0,
        ]
    )
    assert_allclose(time_points, time_points_expected, atol=1e-3, rtol=1e-3)

    assert expression_data_from_gnw.genes == [
        "Node_0",
        "Node_1",
        "Node_2",
        "Node_3",
        "Node_4",
        "Node_5",
        "Node_6",
        "Node_7",
        "Node_8",
        "Node_9",
    ]

    # For each knocked-out index check that it also has an expression of 0 at the end.
    for env_vector, expressions in expression_data_from_gnw:
        ko_gene_index = (env_vector == 1).nonzero().squeeze()
        assert_close(
            expressions[ko_gene_index][-1], torch.tensor(0.0), rtol=1e-4, atol=1e-4
        )


def test_match_genes_to_list(expression_data_from_gnw):
    expression_data_from_gnw.match_genes_to_list(
        [
            "Node_1",
            "Node_0",
            "Node_2",
            "Node_3",
            "Node_5",
            "Node_4",
            "Node_6",
            "Node_7",
            "Node_8",
            "Node_9",
        ]
    )
    assert expression_data_from_gnw.genes == [
        "Node_1",
        "Node_0",
        "Node_2",
        "Node_3",
        "Node_5",
        "Node_4",
        "Node_6",
        "Node_7",
        "Node_8",
        "Node_9",
    ]
    env_tensor, expression = expression_data_from_gnw[0]
    assert_close(
        env_tensor, torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), check_dtype=False
    )

    initial_states = expression[:, 0]
    assert_close(
        initial_states,
        torch.tensor(
            [
                0.6281,
                0.1093,
                0.7730,
                0.9152,
                0.6522,
                0.5725,
                0.3622,
                0.0797,
                0.7164,
                0.4301,
            ]
        ),
        atol=1e-3,
        rtol=1e-3,
    )


def test_scale_time_series(expression_data_from_gnw):
    assert np.allclose(
        expression_data_from_gnw.get_time_points(),
        np.array(
            [
                0.0,
                50.0,
                100.0,
                150.0,
                200.0,
                250.0,
                300.0,
                350.0,
                400.0,
                450.0,
                500.0,
                550.0,
                600.0,
                650.0,
                700.0,
                750.0,
                800.0,
                850.0,
                900.0,
                950.0,
                1000.0,
            ]
        ),
    )

    expression_data_from_gnw.scale_time_series()
    assert np.allclose(
        expression_data_from_gnw.get_time_points(),
        np.array(
            [
                0.0,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ]
        ),
    )
