import numpy as np

from pipeline.base import BaseReducer


class TSNEReducer(BaseReducer):
    """Dimensionality reduction via t-SNE."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("t-SNE reducer not yet implemented.")
