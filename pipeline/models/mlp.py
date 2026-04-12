import numpy as np

from pipeline.base import BaseModel


class MLPModel(BaseModel):
    """Multi-layer perceptron classifier (required for LRP attribution)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("MLP model not yet implemented.")

    @property
    def model(self):
        raise NotImplementedError("MLP model not yet implemented.")
