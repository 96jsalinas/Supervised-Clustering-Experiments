import numpy as np

from pipeline.base import BaseAttributor, BaseModel


class LIMEAttributor(BaseAttributor):
    """LIME-based local attributions."""

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, model: BaseModel
    ) -> np.ndarray:
        raise NotImplementedError("LIME attributor not yet implemented.")
