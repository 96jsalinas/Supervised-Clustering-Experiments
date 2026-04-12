import numpy as np
from sklearn.cluster import DBSCAN

from pipeline.base import BaseClusterer


class DBSCANClusterer(BaseClusterer):
    """Density-based clustering via DBSCAN."""

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        params = self.config.get("params", {})
        return DBSCAN(**params).fit_predict(X)
