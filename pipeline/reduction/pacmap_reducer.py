import numpy as np
import pacmap

from pipeline.base import BaseReducer


class PaCMAPReducer(BaseReducer):
    """Dimensionality reduction via PaCMAP."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        random_state = self.config.get("random_state", None)
        params = self.config.get("params", {})
        reducer = pacmap.PaCMAP(random_state=random_state, **params)
        return reducer.fit_transform(np.asarray(X, dtype=np.float32))
