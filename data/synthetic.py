import numpy as np
from sklearn.datasets import make_classification

# Keys handled explicitly; everything else is forwarded to make_classification.
_RESERVED_KEYS = {"n_samples", "n_classes", "n_clusters_per_class", "shuffle",
                  "random_state"}


def generate_data(
    data_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper around sklearn.make_classification.

    Returns (X, y_class, y_subcluster) where y_subcluster encodes the
    true subcluster identity (class * n_clusters_per_class + cluster_index).

    Any config keys not in _RESERVED_KEYS are forwarded directly to
    make_classification as keyword arguments (e.g. shift, scale, flip_y,
    hypercube, class_sep).

    IMPORTANT: This relies on shuffle=False so that samples are generated
    in deterministic (class, cluster) order. If shuffle is True the
    subcluster labels will be wrong.
    """
    n_samples = data_config["n_samples"]
    n_classes = data_config["n_classes"]
    n_clusters_per_class = data_config["n_clusters_per_class"]
    shuffle = data_config.get("shuffle", False)

    if shuffle:
        raise ValueError(
            "shuffle must be False to recover true subcluster labels. "
            "Set shuffle: false in the config."
        )

    extra_kwargs = {k: v for k, v in data_config.items() if k not in _RESERVED_KEYS}

    X, y = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        shuffle=False,
        random_state=data_config.get("random_state", None),
        **extra_kwargs,
    )

    # Reconstruct subcluster labels from the deterministic ordering.
    # make_classification with shuffle=False allocates samples to
    # (n_classes * n_clusters_per_class) groups in order, distributing
    # n_samples // total_clusters samples per group (remainders go to
    # the first few groups).
    total_clusters = n_classes * n_clusters_per_class
    base_size = n_samples // total_clusters
    remainder = n_samples % total_clusters

    y_subcluster = np.empty(n_samples, dtype=int)
    offset = 0
    for cluster_idx in range(total_clusters):
        size = base_size + (1 if cluster_idx < remainder else 0)
        y_subcluster[offset : offset + size] = cluster_idx
        offset += size

    return X, y, y_subcluster
