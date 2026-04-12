import numpy as np
from sklearn.cluster import HDBSCAN

from pipeline.base import BaseClusterer


class HDBSCANClusterer(BaseClusterer):
    """Hierarchical density-based clustering via HDBSCAN."""

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        params = self.config.get("params", {})
        return HDBSCAN(**params).fit_predict(X)
