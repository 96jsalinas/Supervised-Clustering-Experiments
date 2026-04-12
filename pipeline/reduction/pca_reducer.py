import numpy as np

from pipeline.base import BaseReducer


class PCAReducer(BaseReducer):
    """Dimensionality reduction via PCA."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PCA reducer not yet implemented.")
