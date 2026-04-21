import numpy as np
from sklearn.cluster import KMeans

from pipeline.base import BaseClusterer


class KMeansClusterer(BaseClusterer):
    """Centroid-based clustering via k-means."""

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        random_state = self.config.get("random_state", None)
        params = self.config.get("params", {})
        return KMeans(random_state=random_state, **params).fit_predict(X)
