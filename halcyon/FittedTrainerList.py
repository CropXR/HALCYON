from textwrap import fill
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


import logging
from typing import Callable
from typing_extensions import Self
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from natsort import natsorted

from halcyon.TidyDfResults import TidyDfResults
from halcyon.ModelRunner import ModelRunner


class FittedTrainerList:
    def __init__(
        self,
        trainer_list: list[ModelRunner],
        model_name_func: Callable[[pd.Series], str] | None = None,
    ):
        """Object will be a list of training jobs that have been run and can be used to summarise results. Requires that saved model path is present"""
        # Ensure first model in list has most genes. Needed for drawing the ground truth in some plots
        self.trainer_list = sorted(
            trainer_list, key=lambda t: len(t.dataset.genes), reverse=True
        )
        # Default naming function if none is provided
        self.model_to_name_func = model_name_func or (
            lambda summary: f"{summary['model_cls']}_hidden_{summary.get('hidden_size', 0)}"
        )

    @classmethod
    def load_saved_jobs(cls, train_jobs: list[ModelRunner]) -> Self:
        results = []
        for job in train_jobs:
            try:
                job.load(job.save_path)
                results.append(job)
            except FileNotFoundError as e:
                logging.error(f"Error loading model for job {job.save_path}: {e}")
        return cls(results)

    def get_all_genes(self) -> list[tuple[str]]:
        """Returns list of all genes in all trainers. Starts with genes that are intersection, and then the ones that are disjoint"""
        all_genes = [set(trainer.dataset.genes) for trainer in self.trainer_list]
        intersection = set.intersection(*all_genes)
        other_genes = set.union(*all_genes) - intersection
        allgenes = natsorted(intersection) + natsorted(other_genes)
        return [(i,) for i in allgenes]

    def get_double_ko_combinations(self) -> list[tuple[str]]:
        double_ko_list = []
        for trainer in self.trainer_list:
            double_ko_data = trainer.double_ko_dataset
            for condition in double_ko_data.conditions:
                double_ko_list.append(condition.ko_gene)
        # Remove duplicates
        return list(set(double_ko_list))

    def _get_gene_colourmap(self):
        """Used to consistently map gene to same colour"""
        all_genes = self.get_all_genes()
        colour_map = dict(
            zip(
                [gene[0] for gene in all_genes],
                sns.color_palette("tab20", n_colors=len(all_genes)),
            )
        )
        return colour_map

    def plot_double_ko(self):
        # Get dataset from trainer object
        dataset_getter = lambda trainer: trainer.double_ko_dataset
        conditions_to_map = self.get_double_ko_combinations()
        self.plot_all_model_fits_for_specified_kos(conditions_to_map, dataset_getter)

    def plot_single_ko(self):
        # Single KOs so conditions of knockouts should be each gene
        dataset_getter = lambda trainer: trainer.dataset
        conditions_to_map = self.get_all_genes()
        return self.plot_all_model_fits_for_specified_kos(
            conditions_to_map, dataset_getter
        )

    def predict_tidy_dataframe_for_all_models_specific_kos(
        self,
        kos_to_predict: list[tuple[str]],
        dataset_getter: Callable,
    ) -> TidyDfResults:
        # Create tidy dataframe
        out_records = []
        for trainer in self.trainer_list:
            # Get the correct GT dataset
            dataset: ExpressionDataWithConditions = dataset_getter(trainer)
            # TODO add the network info?
            # Calculate the predictions per model.
            logging.info(f"Predicting for: {trainer}")
            new_predictions = trainer._tidy_df_of_predictions_on_dataset(
                dataset, kos_to_predict
            )
            out_records.extend(new_predictions)

        return TidyDfResults(pd.concat(out_records))

    def plot_all_model_fits_for_specified_kos(
        self,
        kos_to_plot: list[tuple[str]],
        dataset_getter: Callable,
        first_n_results: int = None,
    ):
        if first_n_results:
            kos_to_plot = kos_to_plot[:first_n_results]

        tidy_df = self.predict_tidy_dataframe_for_all_models_specific_kos(
            kos_to_plot, dataset_getter
        )
        return tidy_df.plot_model_plots()
