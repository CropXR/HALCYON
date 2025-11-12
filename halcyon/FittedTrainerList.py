import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


import logging
from typing import Callable
from typing_extensions import Self
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from natsort import natsorted

from halcyon.Trainer import Trainer


class FittedTrainerList:
    def __init__(
        self,
        trainer_list: list[Trainer],
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
    def from_train_jobs(cls, train_jobs: list[Trainer]) -> Self:
        results = []
        for job in train_jobs:
            try:
                job.load_if_exists()
                results.append(job)
            except FileNotFoundError as e:
                logging.error(f"Error loading model for job {job.save_path}: {e}")
        return cls(results)

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([i.summary() for i in self.trainer_list])

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

    def plot_all_model_fits_into_one_fig(
        self,
        first_n_results: int = None,
        only_double_ko: bool = False,
        ymin: float = 0,
        ymax: float = 1.1,
    ):
        all_genes = self.get_all_genes()
        if only_double_ko:
            conditions_to_map = self.get_double_ko_combinations()
        else:
            # Single knockouts
            conditions_to_map = all_genes

        if first_n_results:
            conditions_to_map = conditions_to_map[:first_n_results]

        condition_to_row_map = {
            condition: row_index
            for row_index, condition in enumerate(conditions_to_map)
        }
        colour_map = dict(
            zip(
                [g[0] for g in all_genes],
                sns.color_palette("tab20", n_colors=len(all_genes)),
            )
        )
        n_cols = len(self.trainer_list) + 1
        n_rows = len(condition_to_row_map)
        fig, axs = plt.subplots(
            n_rows, n_cols, sharey=True, sharex="col", figsize=(4 * n_cols, 3 * n_rows)
        )

        for column_index, trainer in enumerate(self.trainer_list):
            if only_double_ko:
                dataset = trainer.double_ko_dataset
            else:
                dataset = trainer.dataset

            for ko_gene, row_index in condition_to_row_map.items():
                try:
                    condition, expressions = (
                        dataset._get_condition_and_expression_under_ko_gene(ko_gene)
                    )
                except KeyError:
                    # Condition not predicted by model
                    continue
                env_tensor, expressions_tensor = (
                    condition.as_tensor(),
                    expressions.as_tensor(),
                )
                env_tensor = env_tensor.to(trainer.model.device)
                expressions_tensor = expressions_tensor.to(trainer.model.device)

                if column_index == 0:
                    # Plot ground truth in first column
                    ko_title = f"Ground truth (KO: {', '.join(condition.ko_gene)})"
                    trainer._plot_from_expressions(
                        expressions=expressions_tensor.detach().cpu().numpy(),
                        title=ko_title,
                        axs=axs[row_index, 0],
                        show_legend=False,
                        palette=colour_map,
                    )
                if row_index == 0:
                    # First row should have titles
                    model_title = trainer.get_save_file_name()[:-3]
                else:
                    model_title = None
                trainer._plot_one_prediction(
                    expressions_tensor[:, 0].unsqueeze(0),
                    model_title,
                    env_tensor,
                    axs[row_index, column_index + 1],
                    show_legend=False,
                    palette=colour_map,
                )
        for ax in np.ravel(axs):
            ax.set_ylim(ymin, ymax)
        return fig

    def _get_final_loss_heatmap_data(self, column_name: str = "nr_genes"):
        df = self.summary_dataframe()
        df["model_overall_name"] = df.apply(self.model_to_name_func, axis=1)
        # Pivot so genes = columns, modules = rows
        heatmap_data = df.pivot_table(
            index="model_overall_name",
            columns=column_name,
            values="final_loss",
        )
        return heatmap_data

    def final_loss_heatmap(self, column_name: str = "nr_genes", ax=None):
        """Plot heatmap of showing the final loss for each fitted model

        Args:
            column_name (str, optional): Column to group simulations by. Defaults to "nr_genes".
        """
        heatmap_data = self._get_final_loss_heatmap_data(column_name)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        sns.heatmap(
            heatmap_data,
            annot=True,
            ax=ax,
            fmt=".8f",
            cmap="viridis",
            cbar_kws={"label": "Final Loss"},
        )
        ax.set_title("Final Loss Heatmap")

        return fig, ax
