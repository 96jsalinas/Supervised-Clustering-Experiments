import numpy as np

from pipeline.base import BaseAttributor, BaseModel


class LRPAttributor(BaseAttributor):
    """Layer-wise Relevance Propagation via a neural network."""

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, model: BaseModel
    ) -> np.ndarray:
        raise NotImplementedError("LRP attributor not yet implemented.")
