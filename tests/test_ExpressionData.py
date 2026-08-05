from copy import deepcopy

import numpy as np
import torch

from torch.testing import assert_close
from numpy.testing import assert_allclose

from halcyon.Datasets import ExpressionDataWithConditions


def test_extract_info(expression_data_from_gnw: ExpressionDataWithConditions):
    # First sample, expression data, all genes, timepoint 0

    env_tensor, expression, ko_string = expression_data_from_gnw[0]
    assert_close(
        env_tensor, torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), check_dtype=False
    )
    assert ko_string == "KO: Node_0. Env: None"

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
    for env_vector, expressions, _ in expression_data_from_gnw:
        ko_gene_index = (env_vector == 1).nonzero().squeeze()
        assert_close(
            expressions[ko_gene_index][-1], torch.tensor(0.0), rtol=1e-4, atol=1e-4
        )


def test_match_genes_to_list(expression_data_from_gnw: ExpressionDataWithConditions):
    expression_data_with_matching_genes = expression_data_from_gnw.match_genes_to_list(
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
    assert expression_data_with_matching_genes.genes == [
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
    env_tensor, expression, ko_string = expression_data_with_matching_genes[0]
    assert_close(
        env_tensor, torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), check_dtype=False
    )
    assert ko_string == "KO: Node_0. Env: None"

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


def test_split_into_train_and_validation(
    expression_data_from_gnw: ExpressionDataWithConditions,
):
    train, validation = expression_data_from_gnw.split_train_test(test_fraction=0.2)
    assert len(train.genes) == 10
    assert len(validation.genes) == 10
    assert len(train) == 8
    assert len(validation) == 2
    assert set(train.conditions).isdisjoint(set(validation.conditions))


def test_scale_time_series(expression_data_from_gnw: ExpressionDataWithConditions):
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


def test_subsample_time_points(expression_data_from_gnw: ExpressionDataWithConditions):
    expression_data_from_gnw.scale_time_series()
    original_time_points = expression_data_from_gnw.get_time_points()
    assert np.allclose(
        original_time_points,
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
        ],
    )
    unmodified_df = deepcopy(expression_data_from_gnw)
    # Check the 0.5 specifically
    expression_data_from_gnw.subsample_time_points(fraction=0.5)
    subsampled_time_points = expression_data_from_gnw.get_time_points()
    assert np.allclose(
        subsampled_time_points, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    _, expression_data, _ = expression_data_from_gnw[1]
    assert expression_data.shape[1] == len(subsampled_time_points)

    # Some edge cases
    for fraction, expected_length in [(0.4, 9), (0.75, 16)]:
        subsample_expression_df = deepcopy(unmodified_df)
        subsample_expression_df.subsample_time_points(fraction=fraction)
        subsampled_time_points = subsample_expression_df.get_time_points()
        assert 0.0 in subsampled_time_points
        assert len(subsampled_time_points) == expected_length
