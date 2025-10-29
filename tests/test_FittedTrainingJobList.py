import pandas as pd
import pytest
from halcyon.FittedTrainingJobList import FittedTrainingJobList
from halcyon.Models.CellBox import CellBox
from halcyon.Models.NeuralODE import PhoenixAddAndProdLayers
from halcyon.Trainer import TrainingJob
from tests.test_GNWSimRunner import sim_result
from pandas.testing import assert_frame_equal


@pytest.fixture
def train_jobs(sim_result):
    train_jobs: list[TrainingJob] = []
    train_steps = 2

    train_jobs.extend(
        [
            TrainingJob(
                PhoenixAddAndProdLayers,
                sim_result,
                nr_training_steps=train_steps,
                model_kwargs={"hidden_size": 2},
                optimizer_kwargs={"lr": 0.1},
            )
        ]
    )

    train_jobs.extend(
        [
            TrainingJob(
                CellBox,
                sim_result,
                nr_training_steps=train_steps,
                model_kwargs={"self_activations": True},
                optimizer_kwargs={"lr": 0.1},
            )
        ]
    )

    len(train_jobs)
    for train_job in train_jobs:
        train_job.save_path = (
            train_job.save_path.parent.parent
            / "expected_output"
            / train_job.save_path.name
        )
    return train_jobs


@pytest.fixture
def result_list_from_jobs(train_jobs):
    result_list = FittedTrainingJobList.from_train_jobs(train_jobs)
    return result_list


def test_check_result_list(result_list_from_jobs):
    assert len(result_list_from_jobs) == 2


def test_final_loss_heatmap(result_list_from_jobs):
    # At least checks if no errors during plot creation
    out_df = result_list_from_jobs._get_final_loss_heatmap_data()
    gt_df = pd.DataFrame(
        {
            10: {
                "CellBox_hidden_nan": 0.06147445738315582,
                "PhoenixAddAndProdLayers_hidden_2.0": 0.060300204902887344,
            }
        },
    )
    gt_df.index.name = "model_overall_name"
    gt_df.columns.name = "nr_genes"
    assert_frame_equal(out_df, gt_df)
    result_list_from_jobs.final_loss_heatmap()


def test_plot_double_ko(result_list_from_jobs):
    result_list_from_jobs.plot_all_model_fits_into_one_fig(only_double_ko=True)
