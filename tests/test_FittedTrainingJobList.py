import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import MagicMock
from halcyon.FittedTrainerList import FittedTrainerList


def test_check_result_list(result_list_from_jobs: FittedTrainerList):
    assert len(result_list_from_jobs.trainer_list) == 4


def test_combine_genes():
    trainer_1 = MagicMock()
    trainer_1.dataset.genes = ["G2", "G9", "G6"]

    trainer_2 = MagicMock()
    trainer_2.dataset.genes = [f"G{i}" for i in range(10)]
    trainer_list = FittedTrainerList([trainer_1, trainer_2])
    assert trainer_list.get_all_genes() == [
        ("G2",),
        ("G6",),
        ("G9",),
        ("G0",),
        ("G1",),
        ("G3",),
        ("G4",),
        ("G5",),
        ("G7",),
        ("G8",),
    ]


def test_final_loss_heatmap(result_list_from_jobs: FittedTrainerList):
    # At least checks if no errors during plot creation
    out_df = result_list_from_jobs._get_final_loss_heatmap_data()
    expected_df = pd.DataFrame(
        {
            10: {
                "CellBox_hidden_nan": 0.031171953957527875,
                "HybridModel_hidden_2.0": 0.012585,
                "PhoenixAddAndProdLayers_hidden_2.0": 0.05473248716443777,
                "PhoenixOnlyAddLayers_hidden_2.0": 0.049757883930578825,
            }
        }
    )
    expected_df.index.name = "model_overall_name"
    expected_df.columns.name = "nr_genes"
    assert_frame_equal(out_df, expected_df, atol=1e-3, rtol=1e-3)
    result_list_from_jobs.final_loss_heatmap()


def test_plot_double_ko(result_list_from_jobs: FittedTrainerList):
    fig = result_list_from_jobs.plot_all_model_fits_into_one_fig(only_double_ko=True)
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == [
        "Ground truth (KO: Node_0, Node_4)",
        "CellBox phi tanh self\nactivations True lr 0.1",
        "PhoenixOnlyAddLayers hidden\nsize 2 synthesis only positive\nFalse lr 0.1",
        "PhoenixAddAndProdLayers hidden\nsize 2 synthesis only positive\nFalse lr 0.1",
        "HybridModel hidden size 2\nsynthesis only positive False\nlr 0.1",
        "Ground truth (KO: Node_1, Node_6)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_2, Node_3)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_1, Node_5)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_0, Node_2)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_0, Node_3)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_0, Node_7)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_2, Node_4)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_3, Node_4)",
        "",
        "",
        "",
        "",
    ]


def test_plot_single_ko(result_list_from_jobs):
    fig = result_list_from_jobs.plot_all_model_fits_into_one_fig()
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == [
        "Ground truth (KO: Node_0)",
        "CellBox phi tanh self\nactivations True lr 0.1",
        "PhoenixOnlyAddLayers hidden\nsize 2 synthesis only positive\nFalse lr 0.1",
        "PhoenixAddAndProdLayers hidden\nsize 2 synthesis only positive\nFalse lr 0.1",
        "HybridModel hidden size 2\nsynthesis only positive False\nlr 0.1",
        "Ground truth (KO: Node_1)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_2)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_3)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_4)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_5)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_6)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_7)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_8)",
        "",
        "",
        "",
        "",
        "Ground truth (KO: Node_9)",
        "",
        "",
        "",
        "",
    ]
