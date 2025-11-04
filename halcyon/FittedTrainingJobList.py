from halcyon.Trainer import TrainingJob


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


import logging
from collections import UserList
from typing import Callable, Generic, TypeVar
from typing_extensions import Self

T = TypeVar("T", bound="TrainingJob")


class FittedTrainingJobList(UserList[T], Generic[T]):
    def __init__(
        self,
        data: list[T] | None = None,
        model_name_func: Callable[[pd.Series], str] | None = None,
    ):
        """Object will be a list of training jobs that have been run and can be used to summarise results. Requires that saved model path is present"""
        super().__init__(data or [])
        # Default naming function if none is provided
        self.model_to_name_func = model_name_func or (
            lambda summary: f"{summary['model_cls']}_hidden_{summary.get('hidden_size', 0)}"
        )

    @classmethod
    def from_train_jobs(cls, train_jobs: list[TrainingJob]) -> Self:
        results = []
        for job in train_jobs:
            try:
                job.load_if_exists()
                results.append(job)
            except FileNotFoundError as e:
                logging.error(f"Error loading model for job {job.save_path}: {e}")
        return cls(results)

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([i.summary() for i in self.data])

    def __getitem__(self, index):
        """Copy numpy-like indexing, where you can select mulitple items via a list or array of indices."""
        if isinstance(index, list) or isinstance(index, np.ndarray):
            return FittedTrainingJobList([self.data[i] for i in index])
        return super().__getitem__(index)

    def plot_all_model_fits_into_one_fig(
        self,
        top_n_genes: int | None = None,
        model_subset_indices: list[int] | None = None,
        only_double_ko: bool = False,
        ymin: float = 0,
        ymax: float = 1,
    ):
        model_list = (
            self[model_subset_indices]
            if model_subset_indices is not None
            else self.data
        )
        n_cols = len(model_list) + 1
        if top_n_genes:
            n_rows = top_n_genes
        elif only_double_ko:
            n_rows = len(self[0].get_double_ko_files().conditions)
        else:
            n_rows = len(model_list[0].trainer.dataset)
        fig, axs = plt.subplots(
            n_rows, n_cols, sharey=True, sharex="col", figsize=(4 * n_cols, 3 * n_rows)
        )

        for i, train_job in enumerate(model_list):
            # print(train_job.summary()["model_path"])
            trainer = train_job.trainer

            if only_double_ko:
                dataset = train_job.get_double_ko_files()
            else:
                dataset = trainer.dataset

            for j in range(n_rows):
                exp_condition, _ = dataset._get_experiment_and_expression_at_index(j)
                env_tensor, expressions_tensor = dataset[j]
                env_tensor = env_tensor.to(trainer.model.device)
                expressions_tensor = expressions_tensor.to(trainer.model.device)

                if i == 0:
                    # Plot ground truth in first column
                    ko_title = f"Ground truth (KO: {', '.join(exp_condition.ko_gene)})"
                    trainer._from_expressions_to_plot(
                        expressions=expressions_tensor.detach().cpu().numpy(),
                        title=ko_title,
                        axs=axs[j, 0],
                        show_legend=False,
                    )
                if j == 0:
                    # First row should have titles
                    model_title = self.model_to_name_func(train_job.summary())
                else:
                    model_title = None
                trainer._plot_one_prediction(
                    expressions_tensor[:, 0].unsqueeze(0),
                    model_title,
                    env_tensor,
                    axs[j, i + 1],
                    show_legend=False,
                )
        for ax in np.ravel(axs):
            ax.set_ylim(ymin, ymax)
        plt.show()
        return fig

    def _get_final_loss_heatmap_data(self, column_name: str = "nr_genes"):
        df = self.summary_dataframe()
        df["model_overall_name"] = df.apply(self.model_to_name_func, axis=1)
        # Pivot so genes = columns, modules = rows
        heatmap_data = df.pivot_table(
            index="model_overall_name",
            columns=column_name,
            values="final_loss",
            # aggfunc="mean"
        )
        return heatmap_data

    def final_loss_heatmap(self, column_name: str = "nr_genes"):
        """Plot heatmap of showing the final loss for each fitted model

        Args:
            column_name (str, optional): Column to group simulations by. Defaults to "nr_genes".
        """
        heatmap_data = self._get_final_loss_heatmap_data(column_name)
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".8f",
            cmap="viridis",
            cbar_kws={"label": "Final Loss"},
        )
        plt.title("Final Loss Heatmap")
        plt.tight_layout()
        plt.show()
