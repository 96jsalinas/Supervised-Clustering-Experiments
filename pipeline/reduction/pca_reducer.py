import numpy as np
from sklearn.decomposition import PCA

from pipeline.base import BaseReducer


class PCAReducer(BaseReducer):
    """Linear dimensionality reduction via PCA."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        random_state = self.config.get("random_state", None)
        params = self.config.get("params", {})
        return PCA(random_state=random_state, **params).fit_transform(X)
