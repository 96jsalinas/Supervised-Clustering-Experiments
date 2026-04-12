from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

from pipeline.runner import RunResult


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


def save_umap_raw_true_labels(result: RunResult, figures_dir: Path):
    """UMAP projection of raw features, colored by ground truth labels."""
    reducer = UMAP(n_components=2, n_neighbors=200, min_dist=0.0, random_state=42)
    raw_2d = reducer.fit_transform(result.X_raw)

    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter(ax, raw_2d, result.y_class, "UMAP of raw features (true labels)")
    fig.tight_layout()
    fig.savefig(figures_dir / "umap_raw_true_labels.png", dpi=150)
    plt.close(fig)


def save_umap_shap_true_labels(result: RunResult, figures_dir: Path):
    """UMAP projection of SHAP values, colored by ground truth labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter(ax, result.embedding_2d, result.y_class,
             "UMAP of SHAP values (true labels)")
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


def save_shap_importance_bar(result: RunResult, figures_dir: Path):
    """Bar chart of mean absolute SHAP values per feature."""
    mean_abs = np.mean(np.abs(result.attributions), axis=0)
    n_features = len(mean_abs)
    indices = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(n_features), mean_abs[indices])
    ax.set_xlabel("Feature (sorted by importance)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Feature importance (mean absolute SHAP)")
    fig.tight_layout()
    fig.savefig(figures_dir / "shap_importance_bar.png", dpi=150)
    plt.close(fig)


def save_all_figures(result: RunResult, figures_dir: Path):
    """Generate and save all standard figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_umap_raw_true_labels(result, figures_dir)
    save_umap_shap_true_labels(result, figures_dir)
    save_umap_shap_cluster_labels(result, figures_dir)
    save_shap_importance_bar(result, figures_dir)
