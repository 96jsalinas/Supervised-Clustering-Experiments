import numpy as np
import shap as shap_lib

from pipeline.base import BaseAttributor, BaseModel


class SHAPAttributor(BaseAttributor):
    """SHAP attributions computed from any compatible pre-trained model.

    For tree models (LightGBM, etc.) uses shap.Explainer, which auto-selects
    TreeExplainer. For PyTorch nn.Modules uses shap.GradientExplainer with a
    random background sample drawn from X.
    """

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, model: BaseModel
    ) -> np.ndarray:
        target_class = self.config.get("target_class", 1)
        underlying = model.model

        if self._is_torch_module(underlying):
            attributions = self._torch_shap(X, underlying, target_class)
        else:
            attributions = self._generic_shap(X, underlying, target_class)

        assert attributions.shape == (X.shape[0], X.shape[1]), (
            f"Attribution shape {attributions.shape} does not match "
            f"input shape ({X.shape[0]}, {X.shape[1]})"
        )
        return attributions

    @staticmethod
    def _is_torch_module(obj) -> bool:
        try:
            import torch.nn as nn
        except ImportError:
            return False
        return isinstance(obj, nn.Module)

    def _generic_shap(self, X, underlying, target_class):
        explainer = shap_lib.Explainer(underlying)
        values = explainer(X).values
        if values.ndim == 3:
            return values[:, :, target_class]
        if values.ndim == 2:
            return values
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}. Expected 2 or 3 dims."
        )

    def _torch_shap(self, X, net, target_class):
        import torch

        background_size = int(self.config.get("background_size", 100))
        seed = self.config.get("random_state", 0)
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        bg_idx = rng.choice(n, size=min(background_size, n), replace=False)

        X_f = np.asarray(X, dtype=np.float32)
        bg = torch.from_numpy(X_f[bg_idx])
        X_t = torch.from_numpy(X_f)

        net.eval()
        explainer = shap_lib.GradientExplainer(net, bg)
        shap_values = explainer.shap_values(X_t)

        if isinstance(shap_values, list):
            return np.asarray(shap_values[target_class])
        values = np.asarray(shap_values)
        if values.ndim == 3:
            return values[:, :, target_class]
        return values
