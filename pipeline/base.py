from abc import ABC, abstractmethod
import numpy as np


class BaseAttributor(ABC):
    """Transforms raw features into a per-sample attribution matrix."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train a supervised model and compute attributions.

        Returns: attribution matrix of shape (n_samples, n_features).
        """
        ...


class BaseReducer(ABC):
    """Projects an n-dimensional matrix to a lower-dimensional embedding."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Returns: embedding of shape (n_samples, n_components)."""
        ...


class BaseClusterer(ABC):
    """Assigns cluster labels to each row of an input matrix."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Returns: integer label array of shape (n_samples,). -1 = noise."""
        ...
