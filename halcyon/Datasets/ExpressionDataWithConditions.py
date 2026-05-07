import random
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing_extensions import Self

from halcyon.Datasets.ExperimentCondition import ExperimentCondition
from halcyon.Datasets.ExpressionData import ExpressionData
from halcyon.results import DfColumnNames


class ExpressionDataWithConditions(Dataset):
    """Class to store the expression data of multiple experiments output by GeneNetWeaver."""

    def __init__(self, ko_dict: dict[ExperimentCondition, ExpressionData]):
        self._ko_dict: dict[ExperimentCondition, ExpressionData] = ko_dict
        self._check_if_all_genes_same()
        self._check_if_all_timepoints_same()

    def __repr__(self) -> str:
        example_conditions = " & ".join(
            [f"({key.as_string()})" for key in list(self._ko_dict.keys())[:2]]
        )
        return f"ExpressionDataWithConditions with e.g. {example_conditions} and {len(self.genes)} genes"

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

    def subsample_time_points(self, fraction: float):
        """Subsample the time points in the expression data by keeping only a fraction of them. The same time points are kept for each condition."""
        self._check_if_all_timepoints_same()
        for expression_data in self._ko_dict.values():
            expression_data.subsample_time_points(fraction)

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
    def all_environments(self) -> list[str]:
        """Return the list of all environments"""
        return list(list(self._ko_dict.keys())[0].all_environments)

    def scale_time_series(self):
        """Scale time series so it spans from 0 to 1. Can be used to prevent ODEs from exploding."""
        self._check_if_all_timepoints_same()
        for expression_data in self._ko_dict.values():
            expression_data.scale_time_series()

    def split_train_test(self, test_fraction: float) -> tuple[Self, Self]:
        """Split into train and validation sets by randomly splitting knockouts.

        Args:
            test_fraction (float): Fraction of conditions to use for validation (e.g. 0.2 for 20%).

        Returns:
            tuple[Self, Self]: (train, validation) datasets.
        """
        kos_to_pick_from = list(self._ko_dict.keys())
        nr_of_kos_in_test = int(test_fraction * len(self._ko_dict))
        val_keys = set(random.sample(kos_to_pick_from, k=nr_of_kos_in_test))
        train_dict = {k: v for k, v in self._ko_dict.items() if k not in val_keys}
        val_dict = {k: v for k, v in self._ko_dict.items() if k in val_keys}
        return self.__class__(train_dict), self.__class__(val_dict)

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return tuple of tensors: first item KO and environmental condition, and second item the corresponding expression data, third item the condition as a string."""
        experiment_condition, expression_data = (
            self._get_experiment_and_expression_at_index(idx)
        )
        return (
            experiment_condition.as_tensor(),
            expression_data.as_tensor(),
            experiment_condition.as_string(),
        )

    def match_genes_to_list(self, genes: list[str]) -> Self:
        """Reorder the genes in the expression data to match the given list of genes. Return new object with matched genes.

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
        return self.__class__(new_dict)

    def to_long_form_df(self):
        df_list = []
        for experiment_condition, expression_data in self._ko_dict.items():
            df = expression_data.df.melt(
                ignore_index=False,
                var_name=DfColumnNames.TIME_COL,
                value_name=DfColumnNames.GROUND_TRUTH_COL,
            )
            df[DfColumnNames.CONDITION_COL] = experiment_condition.as_string()
            df_list.append(df.reset_index(names=DfColumnNames.GENE_COL))
        return pd.concat(df_list, ignore_index=False)

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
