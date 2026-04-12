import numpy as np
from umap import UMAP

from pipeline.base import BaseReducer


class UMAPReducer(BaseReducer):
    """Dimensionality reduction via UMAP."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        random_state = self.config.get("random_state", None)
        params = self.config.get("params", {})

        reducer = UMAP(random_state=random_state, **params)
        return reducer.fit_transform(X)
