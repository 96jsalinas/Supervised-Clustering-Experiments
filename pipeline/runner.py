from dataclasses import dataclass
import numpy as np

from pipeline.registry import MODELS, ATTRIBUTORS, REDUCERS, CLUSTERERS


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
    """Orchestrates the four-step supervised clustering pipeline:
    model training -> attribution -> dimensionality reduction -> clustering.
    """

    def __init__(self, config: dict):
        model_cfg = config["model"]
        attr_cfg = config["attribution"]
        red_cfg = config["reduction"]
        clust_cfg = config["clustering"]

        for key, registry, label in [
            (model_cfg["method"], MODELS, "model"),
            (attr_cfg["method"], ATTRIBUTORS, "attribution"),
            (red_cfg["method"], REDUCERS, "reduction"),
            (clust_cfg["method"], CLUSTERERS, "clustering"),
        ]:
            if key not in registry:
                raise ValueError(f"Unknown {label} method: '{key}'")

        self.model = MODELS[model_cfg["method"]](model_cfg)
        self.attributor = ATTRIBUTORS[attr_cfg["method"]](attr_cfg)
        self.reducer = REDUCERS[red_cfg["method"]](red_cfg)
        self.clusterer = CLUSTERERS[clust_cfg["method"]](clust_cfg)

    def run(
        self, X: np.ndarray, y_class: np.ndarray, y_subcluster: np.ndarray
    ) -> RunResult:
        """Execute: model training -> attribution -> reduction -> clustering.

        Also clusters directly in the full attribution space (no DR)
        to serve as a comparison baseline.
        """
        print("  Training model...")
        self.model.fit(X, y_class)

        print("  Computing attributions...")
        attributions = self.attributor.fit_transform(X, y_class, self.model)

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
