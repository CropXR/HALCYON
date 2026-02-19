import torch

from halcyon.Models.GeneExpressionModel import GeneExpressionModel


class AlwaysDerivativeZero(GeneExpressionModel):
    def __init__(
        self,
        **kwargs,
    ):
        """A model that always returns zero derivative. Used as a 'baseline' model."""
        super().__init__(**kwargs)
        # Will make optimiser not throw error because it requires a param
        self._dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(state)

    def get_inferred_network(self, **kwargs):
        pass

    def get_hyperparams(self):
        return {}
