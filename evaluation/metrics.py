import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from pipeline.runner import RunResult
from evaluation.classifier import compute_classifier_metrics

# Back-compat alias — older code imported `_compute_classifier` from here
# before the helper moved to `evaluation.classifier`.
_compute_classifier = compute_classifier_metrics


def _compute_external(y_true: np.ndarray, labels: np.ndarray) -> dict:
    """Compute external metrics (require ground truth)."""
    return {
        "ari": adjusted_rand_score(y_true, labels),
        "nmi": normalized_mutual_info_score(y_true, labels),
        "ami": adjusted_mutual_info_score(y_true, labels),
    }


def _compute_internal(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute internal metrics (no ground truth needed).

    Noise points (label == -1) are excluded. If fewer than 2 clusters
    remain after excluding noise, internal metrics are set to NaN.
    """
    mask = labels >= 0
    n_noise = int((~mask).sum())
    X_clean = X[mask]
    labels_clean = labels[mask]

    n_clusters = len(set(labels_clean))
    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }

    return {
        "silhouette": silhouette_score(X_clean, labels_clean),
        "davies_bouldin": davies_bouldin_score(X_clean, labels_clean),
        "calinski_harabasz": calinski_harabasz_score(X_clean, labels_clean),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }


def compute_all_metrics(result: RunResult) -> pd.DataFrame:
    """Compute external and internal metrics for both clustering spaces.

    Returns a DataFrame with two rows: one for the 2D embedding space
    and one for the full attribution space (no DR).
    """
    t = result.timings
    shared_timings = {
        "time_model_fit": t.get("model_fit"),
        "time_attribution": t.get("attribution"),
        "time_reduction": t.get("reduction"),
    }

    y_class_test = (
        result.y_class[result.test_idx] if result.test_idx is not None else None
    )
    classifier = _compute_classifier(y_class_test, result.proba_test)
    n_test = int(len(result.test_idx)) if result.test_idx is not None else 0
    n_train = int(len(result.train_idx)) if result.train_idx is not None else 0

    rows = []
    for space, labels, X_space, t_clust in [
        ("embedding_2d", result.cluster_labels_2d, result.embedding_2d,
         t.get("clustering_2d")),
        ("full_attribution", result.cluster_labels_full, result.attributions,
         t.get("clustering_full")),
    ]:
        external = _compute_external(result.y_subcluster, labels)
        internal = _compute_internal(X_space, labels)
        rows.append({
            "space": space,
            **external,
            **internal,
            **shared_timings,
            "time_clustering": t_clust,
            **classifier,
            "n_train": n_train,
            "n_test": n_test,
        })

    return pd.DataFrame(rows)
