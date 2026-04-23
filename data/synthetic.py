import numpy as np
from sklearn.datasets import make_blobs


def generate_data(
    data_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic classification dataset with known subcluster labels.

    Blobs are generated in an n_informative-dimensional subspace via
    make_blobs, then padded with noise features to reach n_features total.
    Binary class labels are assigned by cluster index modulo n_classes, so
    cluster 0 -> class 0, cluster 1 -> class 1, cluster 2 -> class 0, etc.

    Returns (X, y_class, y_subcluster).

    Config keys
    -----------
    n_samples       : total number of samples
    n_features      : total number of features (informative + noise)
    n_informative   : dimensions in which the blobs are generated
    n_clusters      : number of distinct subclusters
    n_classes       : number of binary class labels (default 2)
    cluster_std     : within-cluster standard deviation (default 1.0)
    center_box      : (min, max) range for cluster center placement in each
                      informative dimension (default (-10, 10)). Reduce this
                      to increase cluster overlap in the raw feature space,
                      e.g. (-2, 2) approximates the overlap level produced
                      by make_classification with class_sep=1.0.
    rotate_informative : if True, multiply the informative subspace output of
                      make_blobs by a random orthogonal matrix Q. Cluster
                      identities, per-cluster sample counts, and all pairwise
                      centroid distances are preserved exactly; only the
                      basis of the informative subspace is randomised. This
                      tests whether results that favour axis-aligned splits
                      (tree ensembles) are an artefact of the feature basis.
                      Default False.
    random_state    : integer seed for reproducibility. Also seeds the
                      rotation matrix when rotate_informative is True.
    """
    n_samples = data_config["n_samples"]
    n_features = data_config["n_features"]
    n_informative = data_config["n_informative"]
    n_clusters = data_config["n_clusters"]
    n_classes = data_config.get("n_classes", 2)
    cluster_std = data_config.get("cluster_std", 1.0)
    random_state = data_config.get("random_state", None)
    rotate_informative = bool(data_config.get("rotate_informative", False))

    center_box = tuple(data_config.get("center_box", (-10.0, 10.0)))

    X_informative, y_subcluster = make_blobs(
        n_samples=n_samples,
        n_features=n_informative,
        centers=n_clusters,
        cluster_std=cluster_std,
        center_box=center_box,
        random_state=random_state,
    )

    if rotate_informative:
        # Randomly rotate the informative subspace.
        #
        # Why: make_blobs produces isotropic Gaussian clusters whose principal
        # axes are parallel to the feature axes. Tree ensembles can split
        # trivially along a coordinate in that regime and win by construction.
        # Multiplying by a random orthogonal matrix Q changes only the basis.
        #
        # What is preserved exactly:
        #   - cluster identity (y_subcluster is untouched; rotation is a bijection)
        #   - per-cluster sample counts
        #   - all pairwise centroid-to-centroid distances (rotation is isometric)
        #   - isotropic within-cluster covariance (QQ^T = I)
        #
        # What changes: cluster principal axes are no longer parallel to the
        # feature axes, so tree splits along a single coordinate no longer
        # align with the informative structure.
        #
        # This is *not* a reimplementation of sklearn.make_classification. We
        # do not change centroid placement, add redundant features, or flip
        # labels. Only the basis of the informative subspace is randomised.
        rng_rot = np.random.default_rng(random_state)
        G = rng_rot.standard_normal((n_informative, n_informative))
        Q, _ = np.linalg.qr(G)
        X_informative = X_informative @ Q

    n_noise = n_features - n_informative
    if n_noise > 0:
        noise_rng = np.random.default_rng(random_state)
        X_noise = noise_rng.standard_normal((n_samples, n_noise))
        X = np.hstack([X_informative, X_noise])
    else:
        X = X_informative

    y_class = (y_subcluster % n_classes).astype(int)

    return X, y_class, y_subcluster
