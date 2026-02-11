import pickle
import torch
from halcyon.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Simulators.NoiseLevel import NoiseLevel

from typing_extensions import Self
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class SimulationResult:
    """Container for storing the results of a simulated gene network.

    Attributes:
        data_path (Path): Directory for output files.
        network_name (str): File name to use for outputting the files.
        nr_genes (int): Number of genes in the network.
        nr_modules (int): Number of modules in the network.
        sparsity (float): How sparse the network is
            (higher values mean fewer connections).
        keep_self_interactions (bool): Whether self-interactions are kept.
        network (RegulatoryNetwork): The generated gene regulatory network.
    """

    data_path: Path
    network_name: str
    nr_genes: int
    nr_modules: int
    sparsity: float
    keep_self_interactions: bool
    network: RegulatoryNetwork
    noise_level: NoiseLevel
    normalise_time_series: bool = True

    def save_to_pkl(self, save_path: Path):
        """Save file to pickle. Can be used to log to MLFlow as an artifact."""
        with save_path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_pkl(cls, load_path: Path) -> Self:
        """Load file from pickle.

        Can be used to load from MLFlow.
        Assumes that all the generated datafiles are in the same directory
        as this pickle file
        """
        with load_path.open("rb") as f:
            data = pickle.load(f)
        data.data_path = load_path.parent
        return data

    def _build_filename(self, filename_end: str) -> Path:
        return (
            self.data_path
            / f"{self.network_name}{self.noise_level.value}{filename_end}"
        )

    def items(self):
        """Iterate over field names and their values (as if this were a dict)"""
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def double_ko_timeseries_file_path(self) -> Path:
        """Get path to double knockout timeseries file."""
        out_path = self._build_filename("dualknockout_timeseries0.tsv")

        assert out_path.exists()
        return out_path

    def get_double_ko_expressiondata(
        self, do_time_scaling: bool = True
    ) -> ExpressionDataWithConditions:
        double_ko_expressions = ExpressionDataWithConditions.from_gnw_double_ko_files(
            expression_file_path=self.double_ko_timeseries_file_path(),
            ko_file_path=self.double_ko_index_file_path(),
        )
        return double_ko_expressions

    def double_ko_index_file_path(self) -> Path:
        """Get path to double knockout index file."""
        out_path = self.data_path / f"{self.network_name}_dualknockouts_indexes.tsv"
        assert out_path.exists()
        return out_path

    def get_ko_timeseries_expressiondata(self) -> ExpressionDataWithConditions:
        """Get knockout timeseries without noise data."""
        out_path = self._build_filename("knockout_timeseries0.tsv")

        assert out_path.exists()

        return ExpressionDataWithConditions.from_genenetweaver_ko_timeseries_output(
            out_path
        )

    def create_datasets(self):
        """Preprocess datasets to be ready for training.

        Returns the single and double KO datasets, as well as the scaling factor that was used to normalise the data over time
            (this can be used to adjust the mechanistic model, which assumes the original time scale)
        """
        single_ko_dataset = self.get_ko_timeseries_expressiondata()
        double_ko_dataset = self.get_double_ko_expressiondata()

        scaling_factor = self._time_scale_all_datasets_and_get_scaling_factor(
            [single_ko_dataset, double_ko_dataset]
        )

        return single_ko_dataset, double_ko_dataset, scaling_factor

    def _time_scale_all_datasets_and_get_scaling_factor(
        self, dataset_list: list[ExpressionDataWithConditions]
    ):
        if self.normalise_time_series:
            previous_scaler = None
            # Do scaling of time series (and supply that to models)
            for my_dataset in dataset_list:
                # Normalise time scale to be from 0 to 1. Helps ODEs not explode
                original_max = max(my_dataset._get_time_points())
                my_dataset.scale_time_series()
                new_max = max(my_dataset._get_time_points())
                scaling_factor = torch.tensor(original_max / new_max)
                if previous_scaler is not None:
                    assert torch.isclose(
                        scaling_factor, previous_scaler
                    ), "Time scaling factors between single and double KO are not identical"
                previous_scaler = scaling_factor
        else:
            scaling_factor = torch.tensor(1.0)
        return scaling_factor
