from dataclasses import dataclass
import numpy as np

from pipeline.registry import ATTRIBUTORS, REDUCERS, CLUSTERERS


@dataclass
class RunResult:
    """Holds all arrays produced by a single pipeline run."""

    X_raw: np.ndarray
    y_class: np.ndarray          # binary class labels (0/1)
    y_subcluster: np.ndarray     # true subcluster identities (0..n_classes*n_clusters-1)
    attributions: np.ndarray
    embedding_2d: np.ndarray
    cluster_labels_2d: np.ndarray
    cluster_labels_full: np.ndarray


class PipelineRunner:
    """Orchestrates the three-step supervised clustering pipeline."""

    def __init__(self, config: dict):
        attr_cfg = config["attribution"]
        red_cfg = config["reduction"]
        clust_cfg = config["clustering"]

        attr_method = attr_cfg["method"]
        red_method = red_cfg["method"]
        clust_method = clust_cfg["method"]

        if attr_method not in ATTRIBUTORS:
            raise ValueError(f"Unknown attribution method: {attr_method}")
        if red_method not in REDUCERS:
            raise ValueError(f"Unknown reduction method: {red_method}")
        if clust_method not in CLUSTERERS:
            raise ValueError(f"Unknown clustering method: {clust_method}")

        self.attributor = ATTRIBUTORS[attr_method](attr_cfg)
        self.reducer = REDUCERS[red_method](red_cfg)
        self.clusterer = CLUSTERERS[clust_method](clust_cfg)

    def run(
        self, X: np.ndarray, y_class: np.ndarray, y_subcluster: np.ndarray
    ) -> RunResult:
        """Execute: attribution -> reduction -> clustering.

        Also clusters directly in the full attribution space (no DR)
        to serve as a comparison baseline.
        """
        print("  Computing attributions...")
        attributions = self.attributor.fit_transform(X, y_class)

        print("  Reducing dimensions...")
        embedding_2d = self.reducer.fit_transform(attributions)

        print("  Clustering in 2D embedding space...")
        cluster_labels_2d = self.clusterer.fit_predict(embedding_2d)

        print("  Clustering in full attribution space (no DR)...")
        cluster_labels_full = self.clusterer.fit_predict(attributions)

        return RunResult(
            X_raw=X,
            y_class=y_class,
            y_subcluster=y_subcluster,
            attributions=attributions,
            embedding_2d=embedding_2d,
            cluster_labels_2d=cluster_labels_2d,
            cluster_labels_full=cluster_labels_full,
        )
