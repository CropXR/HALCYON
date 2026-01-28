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
                "HybridModel_hidden_2.0": 0.006076498026959598,
                "PhoenixAddAndProdLayers_hidden_2.0": 0.07486253883689642,
                "PhoenixOnlyAddLayers_hidden_2.0": 0.06326545365154743,
            }
        }
    )
    expected_df.index.name = "model_overall_name"
    expected_df.columns.name = "nr_genes"
    assert_frame_equal(out_df, expected_df, atol=1e-3, rtol=1e-3)
    result_list_from_jobs.final_loss_heatmap()


def test_generate_tidy_df(result_list_from_jobs: FittedTrainerList):
    dataset_getter = lambda trainer: trainer.dataset
    conditions_to_map = result_list_from_jobs.get_all_genes()
    tidy_df = result_list_from_jobs.predict_tidy_dataframe_for_all_models_specific_kos(
        conditions_to_map, dataset_getter
    )
    # Just check column titles
    assert tidy_df.df.columns.tolist() == [
        tidy_df.GENE_COL,
        tidy_df.TIME_COL,
        tidy_df.EXPRESSION_COL,
        tidy_df.MODEL_COL,
        tidy_df.CONDITION_COL,
    ]
