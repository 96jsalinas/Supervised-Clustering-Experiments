import numpy as np

from pipeline.base import BaseAttributor


class LIMEAttributor(BaseAttributor):
    """LIME-based local attributions."""

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LIME attributor not yet implemented.")
