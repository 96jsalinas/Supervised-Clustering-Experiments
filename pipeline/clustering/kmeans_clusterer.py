import numpy as np

from pipeline.base import BaseClusterer


class KMeansClusterer(BaseClusterer):
    """Centroid-based clustering via k-means."""

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("k-means clusterer not yet implemented.")
