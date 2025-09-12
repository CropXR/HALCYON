from halcyon.Datasets.ExperimentCondition import ExperimentCondition
from halcyon.Datasets.ExpressionData import ExpressionData

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing_extensions import Self

from io import StringIO
from pathlib import Path


class ExpressionDataWithConditions(Dataset):
    """Class to store the expression data of multiple experiments output by GeneNetWeaver."""

    def __init__(self, ko_dict: dict[ExperimentCondition, ExpressionData]):
        self._ko_dict = ko_dict
        self._check_if_all_genes_same()
        self._check_if_all_timepoints_same()

    def _check_if_all_timepoints_same(self):
        # Assert that all timepoints are the same
        assert (
            len(
                set(
                    [
                        tuple(expression_data._get_time_points())
                        for expression_data in self._ko_dict.values()
                    ]
                )
            )
            == 1
        ), "All ExpressionData objects must contain the same time points."

    def get_time_points(self) -> np.ndarray:
        self._check_if_all_timepoints_same()
        return next(iter(self._ko_dict.values()))._get_time_points()

    def _check_if_all_genes_same(self):
        # Assert that all ExpressionData contain the same genes
        assert (
            len(
                set(
                    [
                        tuple(expression_data.genes)
                        for expression_data in self._ko_dict.values()
                    ]
                )
            )
            == 1
        ), "All ExpressionData objects must contain the same genes."

    @property
    def genes(self) -> list:
        """Return the genes in the expression data"""
        self._check_if_all_genes_same()
        return self._ko_dict[list(self._ko_dict.keys())[0]].genes

    @property
    def conditions(self) -> list[ExperimentCondition]:
        """Return the list of ExperimentConditions where each item belongs to one expression dataset"""
        return list(self._ko_dict.keys())

    @property
    def all_environments(self) -> tuple[str, ...]:
        """Return the list of all environments"""
        return list(self._ko_dict.keys())[0].all_environments

    def scale_time_series(self):
        """Scale time series so it spans from 0 to 1. Can be used to prevent ODEs from exploding."""
        self._check_if_all_timepoints_same()
        for expression_data in self._ko_dict.values():
            expression_data.scale_time_series()

    def _get_time_points(self) -> np.ndarray:
        """Return the time points of the samples"""
        self._check_if_all_timepoints_same()
        return self._ko_dict[list(self._ko_dict.keys())[0]]._get_time_points()

    def __len__(self) -> int:
        return len(self._ko_dict)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tuple of first item KO and environmental condition, and second item the corresponding expression data."""
        experiment_condition = list(self._ko_dict.keys())[idx]
        expression_data = self._ko_dict[experiment_condition]
        return experiment_condition.as_tensor(), expression_data.as_tensor()

    def match_genes_to_list(self, genes: list[str]):
        """Reorder the genes in the expression data to match the given list of genes.

        Args:
            genes (list): list of strings of genes that should be in the expression data, and the order in which the models expect them to be.
        """
        new_dict = dict()
        for experiment_condition, expression_data in self._ko_dict.items():
            new_experiment_condition = (
                experiment_condition.return_with_changed_gene_list(genes)
            )
            new_dict[new_experiment_condition] = expression_data.match_genes_to_list(
                genes
            )
        self._ko_dict = new_dict

    @classmethod
    def from_genenetweaver_ko_timeseries_output(cls, input_path: Path) -> Self:
        """Create this object from a GeneNetWeaver KO timeseries output file.
        Assumes that each new block of measurements corresponds to the knockout of the next gene (as is default for GNW).
        Args:
            input_path (Path): path to tsv file.
        """
        ko_dict = {}
        text_blocks = input_path.read_text().strip().split("\n\n")
        headers = text_blocks.pop(0)
        headers = headers.split("\t")[1:]

        for gene_name, block in zip(headers, text_blocks):
            experiment_condition = ExperimentCondition(
                all_genes=tuple(headers),
                ko_gene=gene_name,
                all_environments=("test", "test2"),
                environment=None,
            )
            df = pd.read_csv(StringIO(block), sep="\t", names=headers)
            df = df.T
            ko_dict[experiment_condition] = ExpressionData(df)

        return cls(ko_dict)
