"""Cooper's validation contrast: cluster the raw features without SHAP.

Cooper et al. (2021) validate the SHAP-based pipeline by running the identical
UMAP + HDBSCAN steps directly on the raw symptom data (bypassing the model and
SHAP) and showing it produces far worse, less separable clusters (their
Figures 5-6: silhouette 0.140 vs 0.822, most points in one amorphous cluster).

This helper reproduces that contrast: it takes the same reduction and clustering
config the SHAP pipeline used and applies them to the raw feature subset.
"""

import numpy as np

from pipeline.registry import REDUCERS, CLUSTERERS
from evaluation.metrics import _compute_internal


def compute_raw_baseline(X_subset: np.ndarray, config: dict) -> dict:
    """Run reduction + clustering on raw features (no SHAP) and score them.

    Uses the run's own `reduction` and `clustering` config blocks, so the only
    difference from the SHAP arm is the input matrix. Returns the embedding,
    cluster labels, the clusterer's selected params (if it grid-searched), and
    the internal metrics computed the same way as the SHAP arm.
    """
    reducer = REDUCERS[config["reduction"]["method"]](config["reduction"])
    clusterer = CLUSTERERS[config["clustering"]["method"]](config["clustering"])

    embedding = reducer.fit_transform(X_subset)
    labels = clusterer.fit_predict(embedding)
    internal = _compute_internal(embedding, labels)

    return {
        "embedding": embedding,
        "labels": labels,
        "selected_params": getattr(clusterer, "selected_params_", None),
        **internal,
    }
