import numpy as np
import lightgbm as lgb
import shap

from pipeline.base import BaseAttributor


class SHAPLGBMAttributor(BaseAttributor):
    """SHAP attributions computed from a LightGBM classifier."""

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        target_class = self.config.get("target_class", 1)
        random_state = self.config.get("random_state", None)
        model_params = self.config.get("params", {})

        model = lgb.LGBMClassifier(
            random_state=random_state,
            verbosity=-1,
            **model_params,
        )
        model.fit(X, y)

        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        values = shap_values.values
        if values.ndim == 3:
            # Multiclass output: (n_samples, n_features, n_classes)
            attributions = values[:, :, target_class]
        elif values.ndim == 2:
            # Some SHAP versions return (n_samples, n_features) for binary
            attributions = values
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}. "
                "Expected 2 or 3 dimensions."
            )

        assert attributions.shape == (X.shape[0], X.shape[1]), (
            f"Attribution shape {attributions.shape} does not match "
            f"input shape ({X.shape[0]}, {X.shape[1]})"
        )
        return attributions
