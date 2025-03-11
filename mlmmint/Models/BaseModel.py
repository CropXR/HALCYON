from abc import ABC, abstractmethod

import numpy as np

from mlmmint import GeneNetwork
from mlmmint.ExpressionData import ExpressionData


class BaseModel(ABC):
    @abstractmethod
    def get_derivative(self, state: np.ndarray, t: float):
        """Compute the derivative of the system at a given state and time."""
        pass

    @abstractmethod
    def integrate(self, initial_state: np.ndarray, timespan: list):
        """Numerically integrate the ODE over a given time span."""
        pass

    @abstractmethod
    def fit(self, data: ExpressionData):
        """Fit model to the data"""
        pass
