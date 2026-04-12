from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Trains a supervised model that attribution methods will draw from.

    Separating the model from the attribution method ensures that SHAP and LRP
    (and any future methods) are always evaluated on the same trained model,
    making comparisons between attribution methods fair.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on (X, y). Stores the fitted model internally."""
        ...

    @property
    @abstractmethod
    def model(self):
        """Return the underlying fitted model object."""
        ...


class BaseAttributor(ABC):
    """Transforms raw features into a per-sample attribution matrix.

    Receives a pre-trained BaseModel rather than training its own, so that
    multiple attribution methods can be evaluated on identical models.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, model: BaseModel
    ) -> np.ndarray:
        """Compute attributions using the pre-trained model.

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
