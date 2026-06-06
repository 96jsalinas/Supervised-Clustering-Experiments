from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

from pipeline.runner import RunResult


def _umap_n_neighbors(config, n: int) -> int:
    """UMAP n_neighbors for the raw-feature reference plot.

    Reads the run's reduction config when available (so the reference embedding
    matches the pipeline's UMAP settings), falls back to the historical default
    of 200, and clamps below the sample count UMAP requires.
    """
    default = 200
    if config:
        default = (
            config.get("reduction", {}).get("params", {})
            .get("n_neighbors", default)
        )
    return max(2, min(int(default), n - 1))


def _feature_ticks(ax, feature_names, order):
    """Label the x-axis with feature names (in the given order) when known.

    feature_names is None on synthetic runs, where integer indices are kept.
    """
    if feature_names is None:
        return
    names = [feature_names[i] for i in order]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)


def _scatter(ax, embedding: np.ndarray, labels: np.ndarray, title: str):
    """Draw a labeled scatter plot on the given axes."""
    unique = np.unique(labels)
    for lbl in unique:
        mask = labels == lbl
        marker = "x" if lbl == -1 else "o"
        alpha = 0.3 if lbl == -1 else 0.6
        name = "noise" if lbl == -1 else str(lbl)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=name,
            marker=marker,
            alpha=alpha,
            s=10,
        )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=2, fontsize=7, loc="best")


def save_umap_raw_true_labels(result: RunResult, figures_dir: Path, config=None):
    """UMAP of the raw features for the clustered rows, colored by class.

    Restricted to the clustered subset so it lines up with the SHAP embedding;
    on a positives-only run this is a single class.
    """
    mask = result._mask()
    X_sub = result.X_raw[mask]
    n_neighbors = _umap_n_neighbors(config, len(X_sub))
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.0,
                   random_state=42)
    raw_2d = reducer.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter(ax, raw_2d, result.subset_y_class(),
             "UMAP of raw features (class labels)")
    fig.tight_layout()
    fig.savefig(figures_dir / "umap_raw_true_labels.png", dpi=150)
    plt.close(fig)


def save_umap_shap_true_labels(result: RunResult, figures_dir: Path):
    """UMAP projection of SHAP values, colored by class label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter(ax, result.embedding_2d, result.subset_y_class(),
             "UMAP of SHAP values (class labels)")
    fig.tight_layout()
    fig.savefig(figures_dir / "umap_shap_true_labels.png", dpi=150)
    plt.close(fig)


def save_umap_shap_cluster_labels(result: RunResult, figures_dir: Path):
    """UMAP projection of SHAP values, colored by cluster assignment."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter(ax, result.embedding_2d, result.cluster_labels_2d,
             "UMAP of SHAP values (cluster labels)")
    fig.tight_layout()
    fig.savefig(figures_dir / "umap_shap_cluster_labels.png", dpi=150)
    plt.close(fig)


def save_umap_shap_subcluster_labels(result: RunResult, figures_dir: Path):
    """2D embedding of attributions, colored by the true subcluster identity.

    Uses y_subcluster (all n_classes x n_clusters labels) rather than the
    binary class label, so visual separation per subcluster is legible. Skipped
    on real data, which has no ground-truth subclusters.
    """
    y_sub = result.subset_y_subcluster()
    if y_sub is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter(ax, result.embedding_2d, y_sub,
             "2D embedding of attributions (true subclusters)")
    fig.tight_layout()
    fig.savefig(figures_dir / "umap_shap_subcluster_labels.png", dpi=150)
    plt.close(fig)


def save_clusters_no_dr_vs_dr(result: RunResult, figures_dir: Path):
    """Side-by-side: clusters discovered in the full attribution space vs
    in the 2D embedding, both plotted on the same 2D embedding coords.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes[0], result.embedding_2d, result.cluster_labels_full,
             "Clusters found in full attribution space")
    _scatter(axes[1], result.embedding_2d, result.cluster_labels_2d,
             "Clusters found in 2D embedding")
    fig.tight_layout()
    fig.savefig(figures_dir / "clusters_no_dr_vs_dr.png", dpi=150)
    plt.close(fig)


def save_per_cluster_shap_profile(result: RunResult, figures_dir: Path,
                                  feature_names=None):
    """Small-multiples bar grid: mean |attribution| per feature, one
    subplot per discovered cluster (from the 2D embedding clustering).

    Features are sorted once by global mean |attribution|; all subplots
    share that ordering so they are directly comparable.
    """
    labels = result.cluster_labels_2d
    attr = result.clustered_attributions()
    unique = [c for c in np.unique(labels) if c != -1]

    if len(unique) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No clusters discovered (all noise)",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(figures_dir / "per_cluster_shap_profile.png", dpi=150)
        plt.close(fig)
        return

    global_importance = np.mean(np.abs(attr), axis=0)
    order = np.argsort(global_importance)[::-1]
    n_features = len(order)

    n = len(unique)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.8 * nrows),
                             squeeze=False, sharey=True)

    for idx, cluster in enumerate(unique):
        ax = axes[idx // ncols][idx % ncols]
        mask = labels == cluster
        cluster_imp = np.mean(np.abs(attr[mask]), axis=0)[order]
        ax.bar(range(n_features), cluster_imp)
        ax.set_title(f"Cluster {cluster} (n={int(mask.sum())})")
        ax.set_xlabel("Feature (global importance order)")
        ax.set_ylabel("Mean |attribution|")
        _feature_ticks(ax, feature_names, order)

    for j in range(len(unique), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(figures_dir / "per_cluster_shap_profile.png", dpi=150)
    plt.close(fig)


def save_shap_importance_bar(result: RunResult, figures_dir: Path,
                             feature_names=None):
    """Bar chart of mean absolute SHAP values per feature (model-level, full X)."""
    mean_abs = np.mean(np.abs(result.attributions), axis=0)
    n_features = len(mean_abs)
    indices = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(n_features), mean_abs[indices])
    ax.set_xlabel("Feature (sorted by importance)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Feature importance (mean absolute SHAP)")
    _feature_ticks(ax, feature_names, indices)
    fig.tight_layout()
    fig.savefig(figures_dir / "shap_importance_bar.png", dpi=150)
    plt.close(fig)


def save_shap_vs_raw(result: RunResult, raw: dict, sil_shap: float,
                     figures_dir: Path):
    """Side-by-side: SHAP-value clusters vs raw-feature clusters (Cooper Figs 5-6).

    Both panels carry their silhouette so the SHAP arm's superior separation is
    legible at a glance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes[0], result.embedding_2d, result.cluster_labels_2d,
             f"SHAP embedding (silhouette={sil_shap:.3f})")
    _scatter(axes[1], raw["embedding"], raw["labels"],
             f"Raw features, no SHAP (silhouette={raw['silhouette']:.3f})")
    fig.tight_layout()
    fig.savefig(figures_dir / "shap_vs_raw.png", dpi=150)
    plt.close(fig)


def save_hdbscan_silhouette_curve(elbow_df, figures_dir: Path, space_tag: str):
    """Max mean silhouette per cluster count from an HDBSCAN grid search.

    Cooper (2021) Figure 4 analogue. No-op unless the clusterer ran an
    auto_select grid (elbow_df is None for fixed-param HDBSCAN or other methods).
    """
    if elbow_df is None or "n_clusters" not in getattr(elbow_df, "columns", []):
        return
    df = elbow_df[elbow_df["n_clusters"] >= 2]
    if df.empty:
        return
    curve = df.groupby("n_clusters")["silhouette"].max()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(curve.index.astype(int), curve.values)
    best_k = int(curve.idxmax())
    ax.axvline(best_k, color="red", linestyle="--",
               label=f"selected k={best_k}")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Max mean silhouette")
    ax.set_title(f"HDBSCAN grid search ({space_tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"hdbscan_silhouette_curve_{space_tag}.png", dpi=150)
    plt.close(fig)


def save_all_figures(result: RunResult, figures_dir: Path, config=None,
                     feature_names=None):
    """Generate and save all standard figures.

    config and feature_names are optional so synthetic batch callers can keep
    calling save_all_figures(result, figures_dir) unchanged. config tunes the
    raw-reference UMAP; feature_names labels the importance/profile axes on real
    data.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_umap_raw_true_labels(result, figures_dir, config=config)
    save_umap_shap_true_labels(result, figures_dir)
    save_umap_shap_subcluster_labels(result, figures_dir)
    save_umap_shap_cluster_labels(result, figures_dir)
    save_clusters_no_dr_vs_dr(result, figures_dir)
    save_shap_importance_bar(result, figures_dir, feature_names=feature_names)
    save_per_cluster_shap_profile(result, figures_dir,
                                  feature_names=feature_names)
    meta = result.clustering_meta or {}
    for tag in ("2d", "full"):
        save_hdbscan_silhouette_curve(
            meta.get(f"elbow_df_{tag}"), figures_dir, tag
        )
