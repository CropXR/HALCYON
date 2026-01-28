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

    def _get_experiment_and_expression_at_index(
        self, idx: int
    ) -> tuple[ExperimentCondition, ExpressionData]:
        """Get objects of experimental condition and expression data at an index."""
        experiment_condition = list(self._ko_dict.keys())[idx]
        expression_data = self._ko_dict[experiment_condition]
        return experiment_condition, expression_data

    def _get_condition_and_expression_under_ko_gene(
        self, ko_gene: tuple[str]
    ) -> tuple[ExperimentCondition, ExpressionData]:
        """Get objects of experimental condition and expression data at an index."""
        # ugly but works:
        possible_keys = []
        for condition in self._ko_dict.keys():
            if condition.ko_gene == ko_gene:
                possible_keys.append(condition)
        if not len(possible_keys) == 1:
            raise KeyError(
                f"Found {len(possible_keys)} matching conditions found for KO {ko_gene}"
            )
        condition_to_use = possible_keys[0]
        expression_to_use = self._ko_dict[condition_to_use]
        return condition_to_use, expression_to_use

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tuple of tensors: first item KO and environmental condition, and second item the corresponding expression data."""
        experiment_condition, expression_data = (
            self._get_experiment_and_expression_at_index(idx)
        )
        return experiment_condition.as_tensor(), expression_data.as_tensor()

    def match_genes_to_list(self, genes: list[str]):
        """Reorder the genes in the expression data to match the given list of genes.

        Args:
            genes (list): list of strings of genes that should be in the expression data, and the order in which the models expect them to be.
        """
        new_dict = dict()
        for experiment_condition, expression_data in self._ko_dict.items():
            try:
                new_experiment_condition = (
                    experiment_condition.return_with_changed_gene_list(genes)
                )
            except KeyError:
                # If genes are not in dataset (e.g. for mechanistic model), we'll get a keyerror, so skip it
                continue
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
                ko_gene=(gene_name,),
                all_environments=(),
                environment=None,
            )
            df = pd.read_csv(StringIO(block), sep="\t", names=headers).T
            ko_dict[experiment_condition] = ExpressionData(df)

        return cls(ko_dict)

    @classmethod
    def from_gnw_double_ko_files(
        cls, expression_file_path: Path, ko_file_path: Path
    ) -> Self:
        """Create this object from GeneNetWeaver double knockout files.
        Args:
            expression_file_path (Path): path to tsv file with expression data.
            ko_file_path (Path): path to tsv file with double KO gene pairs.
        """
        # Load expression data
        ko_dict = {}
        text_blocks = expression_file_path.read_text().strip().split("\n\n")
        headers = text_blocks.pop(0)
        headers = headers.split("\t")[1:]
        # Read corresponding KO genes from ko_file_path
        all_ko_combo_list = ko_file_path.read_text().strip().split("\n")
        # Remove header
        all_ko_combo_list.pop(0)
        assert len(all_ko_combo_list) == len(
            text_blocks
        ), "Number of KO gene pairs must match number of expression data blocks."

        for block, raw_ko_gene_str in zip(text_blocks, all_ko_combo_list):
            ko_for_this_expression_block = []
            for gene_index in raw_ko_gene_str.split("\t"):
                # Index is one higher than the Node_ name
                ko_for_this_expression_block.append(f"Node_{int(gene_index) - 1}")

            experiment_condition = ExperimentCondition(
                all_genes=tuple(headers),
                ko_gene=tuple(ko_for_this_expression_block),
                all_environments=(),
                environment=None,
            )

            df = pd.read_csv(StringIO(block), sep="\t", names=headers)
            df = df.T
            ko_dict[experiment_condition] = ExpressionData(df)

        return cls(ko_dict)
