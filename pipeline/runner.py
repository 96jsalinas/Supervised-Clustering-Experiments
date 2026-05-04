from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from sklearn.model_selection import train_test_split

from pipeline.registry import MODELS, ATTRIBUTORS, REDUCERS, CLUSTERERS
from evaluation.classifier import fit_and_eval_classifier


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
    timings: dict = field(default_factory=dict)  # wall-clock seconds per step
    # Classifier-level evaluation: computed on a stratified held-out test split.
    # Attribution/reduction/clustering still run on the full X so existing
    # cluster-level results remain directly comparable across runs.
    train_idx: np.ndarray = None
    test_idx: np.ndarray = None
    proba_test: np.ndarray = None        # (n_test, n_classes)
    proba_train: np.ndarray = None       # (n_train, n_classes)
    # Tuning artefacts — populated only when the config has an enabled
    # `model.tune` block; None otherwise.
    tuning_selected: dict = None
    tuning_grid: "pd.DataFrame | None" = None
    # Clustering artefacts — populated when the clusterer exposes extra state
    # after fit_predict (e.g. kneedle selected_k_ and elbow_df_).
    clustering_meta: dict = field(default_factory=dict)


class PipelineRunner:
    """Orchestrates the four-step supervised clustering pipeline:
    model training -> attribution -> dimensionality reduction -> clustering.
    """

    def __init__(self, config: dict):
        self.config = config
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

        A stratified train/test split is drawn first; the model is fitted on
        the train partition only and classifier-level metrics are evaluated on
        the held-out test partition. Attribution, reduction and clustering
        continue to run on the full X so cluster-level results stay aligned
        with historical runs.
        """
        eval_cfg = self.config.get("evaluation", {}) or {}
        test_fraction = float(eval_cfg.get("test_fraction", 0.2))
        split_seed = int(eval_cfg.get(
            "split_seed", self.config.get("model", {}).get("random_state", 42)
        ))

        timings: dict = {}

        n = len(X)
        idx_all = np.arange(n)
        train_idx, test_idx = train_test_split(
            idx_all,
            test_size=test_fraction,
            stratify=y_class,
            random_state=split_seed,
        )

        print(f"  Train/test split: n_train={len(train_idx)} "
              f"n_test={len(test_idx)} (test_fraction={test_fraction}, "
              f"split_seed={split_seed})")

        tuning_selected = None
        tuning_grid = None
        tune_cfg = (self.config.get("model", {}) or {}).get("tune") or {}
        if tune_cfg.get("enabled", False):
            from pipeline.tuning import tune_classifier  # local import avoids
            # pulling sklearn.model_selection into cold start when unused.
            print("  Tuning classifier via stratified CV...")
            model_cfg = self.config["model"]
            t0 = perf_counter()
            tuning_selected, tuning_grid = tune_classifier(
                model_cls=MODELS[model_cfg["method"]],
                base_config=model_cfg,
                tune_cfg=tune_cfg,
                X_train=X[train_idx],
                y_train=y_class[train_idx],
            )
            timings["tuning"] = perf_counter() - t0
            # Merge winner into params and rebuild the model instance.
            winner_params = tuning_selected["params"]
            model_cfg.setdefault("params", {}).update(winner_params)
            self.model = MODELS[model_cfg["method"]](model_cfg)
            print(f"  Tuned params: {winner_params}")
            print(f"  CV {tuning_selected['scoring']}: "
                  f"{tuning_selected['cv_score_mean']:.4f} "
                  f"± {tuning_selected['cv_score_std']:.4f}")

        print("  Training model on train partition...")
        classifier_out = fit_and_eval_classifier(
            self.model,
            X[train_idx],
            y_class[train_idx],
            X[test_idx],
            y_class[test_idx],
        )
        timings["model_fit"] = classifier_out["time_model_fit"]
        proba_train = classifier_out["proba_train"]
        proba_test = classifier_out["proba_test"]

        print("  Computing attributions (full X)...")
        t0 = perf_counter()
        attributions = self.attributor.fit_transform(X, y_class, self.model)
        timings["attribution"] = perf_counter() - t0

        print("  Reducing dimensions...")
        t0 = perf_counter()
        embedding_2d = self.reducer.fit_transform(attributions)
        timings["reduction"] = perf_counter() - t0

        clustering_meta: dict = {}

        print("  Clustering in 2D embedding space...")
        t0 = perf_counter()
        cluster_labels_2d = self.clusterer.fit_predict(embedding_2d)
        timings["clustering_2d"] = perf_counter() - t0
        clustering_meta["selected_k_2d"] = getattr(self.clusterer, "selected_k_", None)
        clustering_meta["elbow_df_2d"] = getattr(self.clusterer, "elbow_df_", None)

        print("  Clustering in full attribution space (no DR)...")
        t0 = perf_counter()
        cluster_labels_full = self.clusterer.fit_predict(attributions)
        timings["clustering_full"] = perf_counter() - t0
        clustering_meta["selected_k_full"] = getattr(self.clusterer, "selected_k_", None)
        clustering_meta["elbow_df_full"] = getattr(self.clusterer, "elbow_df_", None)

        return RunResult(
            X_raw=X,
            y_class=y_class,
            y_subcluster=y_subcluster,
            attributions=attributions,
            embedding_2d=embedding_2d,
            cluster_labels_2d=cluster_labels_2d,
            cluster_labels_full=cluster_labels_full,
            timings=timings,
            train_idx=train_idx,
            test_idx=test_idx,
            proba_test=proba_test,
            proba_train=proba_train,
            tuning_selected=tuning_selected,
            tuning_grid=tuning_grid,
            clustering_meta=clustering_meta,
        )
