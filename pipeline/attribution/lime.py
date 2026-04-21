import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from pipeline.base import BaseAttributor, BaseModel


class LIMEAttributor(BaseAttributor):
    """LIME tabular attributions.

    Config keys:
        target_class   : int    default 1
        num_samples    : int    default 1000   (LIME perturbations per instance)
        random_state   : int    default None
    """

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, model: BaseModel
    ) -> np.ndarray:
        target_class = int(self.config.get("target_class", 1))
        num_samples = int(self.config.get("num_samples", 1000))
        random_state = self.config.get("random_state", None)

        underlying = model.model
        if hasattr(underlying, "predict_proba"):
            predict_fn = underlying.predict_proba
        elif hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            raise TypeError(
                "LIME requires a model with a predict_proba method."
            )

        X_f = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_f.shape

        explainer = LimeTabularExplainer(
            training_data=X_f,
            mode="classification",
            discretize_continuous=False,
            random_state=random_state,
        )

        attributions = np.zeros((n_samples, n_features), dtype=np.float64)
        for i in range(n_samples):
            exp = explainer.explain_instance(
                X_f[i],
                predict_fn,
                labels=(target_class,),
                num_features=n_features,
                num_samples=num_samples,
            )
            for feat_idx, weight in exp.as_map()[target_class]:
                attributions[i, feat_idx] = weight

        return attributions
