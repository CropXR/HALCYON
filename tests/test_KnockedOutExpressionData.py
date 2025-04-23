from pathlib import Path

import numpy as np
from mlmmint.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions


def test_parse_from_genenetweaver_output():
    input_path = Path(
        "tests/grnsimdummydata/InSilicoSize10-Yeast1_knockout_timeseries.tsv"
    )
    ko_expression_data = (
        ExpressionDataWithConditions.from_genenetweaver_ko_timeseries_output(input_path)
    )
    environment, my_expression_data = ko_expression_data[0]
    # assert my_expression_data.genes == [f"G{i}" for i in range(1, 11)]
    # assert environment.ko_gene == 'G1'
    # assert np.allclose(my_expression_data._get_time_points(), range(0,1010,50))
    # assert np.isclose(my_expression_data.df.at['G4', 150], 0.3854203)
    # _, my_expression_data = ko_expression_data[9]
    # assert np.isclose(my_expression_data.df.at['G10', 1000], 0.0000000)
