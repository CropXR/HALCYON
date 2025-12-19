import logging
import torch

from halcyon.Models.MachineLearningModel import MachineLearningModel
from halcyon.Models.MechanisticModel import MechanisticModel
from halcyon.Models.Model import Model
from halcyon.RegulatoryNetwork import RegulatoryNetwork


class HybridModel(Model):
    def __init__(self, mm: MechanisticModel, ml: MachineLearningModel, **kwargs):
        """
        Initialize the HybridModel with a mechanistic model and a machine learning model.
        Args:
            mm (MechanisticModel): The mechanistic model instance.
            ml (Model): The machine learning model instance.
        Attributes:
            mm (MechanisticModel): The mechanistic model instance.
            ml (Model): The machine learning model instance.
            mm_indices (torch.Tensor): Precomputed indices for the mechanistic model genes.
            ml_indices (torch.Tensor): Precomputed indices for the machine learning model genes.
        """
        all_modelled_genes = list(
            set(mm.modelled_gene_labels) | set(ml.modelled_gene_labels)
        )
        assert (
            mm.condition_names == ml.condition_names
        ), f"Conditions of {mm} and {ml} should match"
        super().__init__(
            modelled_gene_labels=list(all_modelled_genes),
            condition_names=mm.condition_names,
            **kwargs,
        )
        self._mm = mm
        self._ml = ml
        for model in [self._mm, self._ml]:
            model.device = self.device
            model.to(self.device)
        # Precompute indices
        self._mm_indices = torch.tensor(
            [self.modelled_gene_labels.index(g) for g in mm.modelled_gene_labels],
            device=self.device,
        )
        self._ml_indices = torch.tensor(
            [self.modelled_gene_labels.index(g) for g in ml.modelled_gene_labels],
            device=self.device,
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Perform forward pass

        Args:
            t (torch.Tensor): Time point to pass to forward pass
            state (torch.Tensor): Gene expressions at that time point. Should be in shape [batch_size, n_genes].

        Returns:
            torch.Tensor: Change in expressions of genes. Again in shape [batch_size, n_genes].
        """
        mm_output = self._get_mm_forward(t, state)
        ml_output = self._get_ml_forward(t, state)

        output = torch.zeros_like(state, device=self.device)
        output.index_add_(dim=1, index=self._mm_indices, source=mm_output)
        output.index_add_(dim=1, index=self._ml_indices, source=ml_output)
        return output

    def set_timescaler(self, new_value: torch.Tensor):
        self._mm.set_timescaler(new_value)

    def _get_ml_forward(self, t, state) -> torch.Tensor:
        ml_input = state[:, self._ml_indices]
        ml_output = self._ml.forward(t, ml_input)
        return ml_output

    def _get_mm_forward(self, t, state) -> torch.Tensor:
        mm_input = state[:, self._mm_indices]
        mm_output = self._mm.forward(t, mm_input)
        return mm_output

    def get_inferred_network(self, threshold=None) -> RegulatoryNetwork:
        """Get the inferred network

        Returns:
            RegulatoryNetwork: Inferred network from data
        """
        logging.warning(
            "Now just use the ML-model network inference, but later on combine this with MM?"
        )
        return self._ml.get_inferred_network(threshold=threshold)

    def set_conditions(self, env_conditions: torch.Tensor):
        for model, gene_indices in zip(
            [self._mm, self._ml], [self._mm_indices, self._ml_indices]
        ):
            # The knockouts
            genes_and_conditions = env_conditions[:, gene_indices]
            if self.n_conditions > 0:
                # Also include the conditions (e.g. heat or drought that are passed as flag)
                condition_flags = env_conditions[:, : self.n_conditions]
                genes_and_conditions = torch.cat(
                    (genes_and_conditions, condition_flags)
                )
            model.set_conditions(genes_and_conditions)

    def get_hyperparams(self) -> list[tuple[str, int | float | str | bool]]:
        """Get hyperparameters, for now only use ML ones as we assume the MM has none

        These are used to e.g. generate the save file path
        """
        return self._ml.get_hyperparams() + self._mm.get_hyperparams()
