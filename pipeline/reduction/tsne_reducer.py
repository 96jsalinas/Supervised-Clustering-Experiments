import numpy as np
from sklearn.manifold import TSNE

from pipeline.base import BaseReducer


class TSNEReducer(BaseReducer):
    """Nonlinear dimensionality reduction via t-SNE."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        random_state = self.config.get("random_state", None)
        params = self.config.get("params", {})
        return TSNE(random_state=random_state, **params).fit_transform(X)
