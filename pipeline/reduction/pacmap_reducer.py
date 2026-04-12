import numpy as np

from pipeline.base import BaseReducer


class PaCMAPReducer(BaseReducer):
    """Dimensionality reduction via PaCMAP."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PaCMAP reducer not yet implemented.")
