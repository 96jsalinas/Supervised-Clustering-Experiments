import numpy as np
import lightgbm as lgb

from pipeline.base import BaseModel


class LightGBMModel(BaseModel):
    """Gradient boosted tree classifier via LightGBM."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = lgb.LGBMClassifier(
            random_state=self.config.get("random_state", None),
            verbosity=-1,
            **self.config.get("params", {}),
        )
        self._model.fit(X, y)

    @property
    def model(self):
        return self._model
