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
    rows = []
    for space, labels, X_space in [
        ("embedding_2d", result.cluster_labels_2d, result.embedding_2d),
        ("full_attribution", result.cluster_labels_full, result.attributions),
    ]:
        external = _compute_external(result.y_subcluster, labels)
        internal = _compute_internal(X_space, labels)
        rows.append({"space": space, **external, **internal})

    return pd.DataFrame(rows)
