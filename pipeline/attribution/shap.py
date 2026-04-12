import numpy as np
import shap as shap_lib

from pipeline.base import BaseAttributor, BaseModel


class SHAPAttributor(BaseAttributor):
    """SHAP attributions computed from any compatible pre-trained model.

    Uses shap.Explainer, which selects the appropriate explainer backend
    (TreeExplainer for tree models, DeepExplainer for neural networks, etc.)
    automatically based on the model type.
    """

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, model: BaseModel
    ) -> np.ndarray:
        target_class = self.config.get("target_class", 1)

        explainer = shap_lib.Explainer(model.model)
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
