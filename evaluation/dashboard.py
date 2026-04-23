"""Cross-run dashboard.

Walks a results directory, loads every metrics.csv + arrays.npz, and emits
comparison artefacts into an output directory. Intended for the thesis LaTeX
to pick up stable filenames.

Usage:
    python -m evaluation.dashboard
    python -m evaluation.dashboard --results-dir results/ --out figures/dashboard/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Higher-is-better columns use a green-rising colormap; the listed columns
# invert that direction so "best = greenest" stays consistent across the
# rendered table and heatmaps.
LOWER_IS_BETTER = {
    "davies_bouldin",
    "classifier_log_loss",
    "n_noise",
    "time_model_fit",
    "time_attribution",
    "time_reduction",
    "time_clustering",
}

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_runs(results_dir: Path) -> pd.DataFrame:
    """Concatenate every run's metrics.csv with a leading `run` column.

    Backfills a `dataset_tag` column of "default" for older runs that predate
    the robustness sweep (which writes the tag at run time).
    """
    frames = []
    for run_dir in sorted(results_dir.iterdir()):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.is_file():
            continue
        df = pd.read_csv(metrics_path)
        df.insert(0, "run", run_dir.name)
        if "dataset_tag" not in df.columns:
            df["dataset_tag"] = "default"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No metrics.csv files found under {results_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def _scatter_into(ax, embedding: np.ndarray, labels: np.ndarray, title: str):
    for lbl in np.unique(labels):
        mask = labels == lbl
        marker = "x" if lbl == -1 else "o"
        alpha = 0.3 if lbl == -1 else 0.6
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   marker=marker, alpha=alpha, s=6)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def save_embedding_grid(results_dir: Path, out_dir: Path) -> None:
    """One scatter subplot per run, colored by discovered cluster labels."""
    run_dirs = [d for d in sorted(results_dir.iterdir())
                if (d / "arrays.npz").is_file()]
    if not run_dirs:
        print("  no arrays.npz files found, skipping embedding_grid")
        return

    n = len(run_dirs)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)

    for idx, run_dir in enumerate(run_dirs):
        ax = axes[idx // ncols][idx % ncols]
        data = np.load(run_dir / "arrays.npz")
        _scatter_into(ax, data["embedding_2d"], data["cluster_labels_2d"],
                      run_dir.name)

    for j in range(len(run_dirs), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("2D embedding of attributions — discovered clusters",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "embedding_grid.png", dpi=150)
    plt.close(fig)


def save_metric_bars(all_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart per external metric (ARI, NMI, AMI) grouped by run.

    Within each run, the two spaces (2D embedding vs full attribution)
    are shown as adjacent bars so the effect of DR is visible at a glance.
    """
    metrics_to_plot = ["ari", "nmi", "ami"]
    spaces = ["embedding_2d", "full_attribution"]

    runs = list(all_metrics["run"].drop_duplicates())
    fig, axes = plt.subplots(len(metrics_to_plot), 1,
                             figsize=(max(8, 0.5 * len(runs)), 9),
                             sharex=True)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    x = np.arange(len(runs))
    bar_w = 0.4

    for ax, metric in zip(axes, metrics_to_plot):
        for offset, space in zip([-bar_w / 2, bar_w / 2], spaces):
            values = []
            for run in runs:
                row = all_metrics[
                    (all_metrics["run"] == run) & (all_metrics["space"] == space)
                ]
                values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            ax.bar(x + offset, values, width=bar_w, label=space)

        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(runs, rotation=45, ha="right")

    fig.suptitle("External metrics per run (embedding vs full attribution)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "metric_bars.png", dpi=150)
    plt.close(fig)


def save_classifier_bars(all_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart per classifier metric, one bar per run.

    Classifier metrics are evaluated on a held-out test split inside the
    runner and are therefore identical across the two clustering spaces,
    so we take the `embedding_2d` row arbitrarily.
    """
    metrics_to_plot = [
        ("classifier_accuracy", "Accuracy", (0.0, 1.0)),
        ("classifier_auc", "AUC", (0.5, 1.0)),
        ("classifier_f1_macro", "F1 (macro)", (0.0, 1.0)),
    ]
    available = [(c, l, yl) for (c, l, yl) in metrics_to_plot
                 if c in all_metrics.columns]
    if not available:
        print("  no classifier_* columns, skipping classifier bars")
        return

    df = all_metrics[all_metrics["space"] == "embedding_2d"].copy()
    runs = list(df["run"].drop_duplicates())

    fig, axes = plt.subplots(len(available), 1,
                             figsize=(max(8, 0.5 * len(runs)), 3 * len(available)),
                             sharex=True)
    if len(available) == 1:
        axes = [axes]

    x = np.arange(len(runs))
    for ax, (col, label, ylim) in zip(axes, available):
        values = [
            float(df[df["run"] == run][col].iloc[0])
            if len(df[df["run"] == run]) and not pd.isna(df[df["run"] == run][col].iloc[0])
            else np.nan
            for run in runs
        ]
        ax.bar(x, values, color="tab:blue")
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(runs, rotation=45, ha="right")
    fig.suptitle("Classifier-level metrics (held-out test split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "classifier_bars.png", dpi=150)
    plt.close(fig)


def save_stability_figure(all_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Strip plot per external metric showing spread across dataset_tags.

    Each subplot: x = method combo (derived by stripping the `<tag>__` prefix
    from the run name), y = metric value, one point per dataset_tag. Only
    runs that share a method combo across multiple dataset_tags contribute.
    Skipped gracefully when fewer than two tags are present.
    """
    if "dataset_tag" not in all_metrics.columns:
        return
    tags = all_metrics["dataset_tag"].unique()
    if len(tags) < 2:
        print("  only one dataset_tag, skipping stability figure")
        return

    df = all_metrics[all_metrics["space"] == "embedding_2d"].copy()
    df["method_combo"] = df["run"].apply(
        lambda r: r.split("__", 1)[1] if "__" in r else r
    )

    metrics_to_plot = ["ari", "nmi", "ami"]
    combos = sorted(df["method_combo"].unique())
    fig, axes = plt.subplots(len(metrics_to_plot), 1,
                             figsize=(max(8, 0.6 * len(combos)),
                                      3 * len(metrics_to_plot)),
                             sharex=True)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    tag_order = sorted(tags)
    tag_to_color = {t: plt.cm.tab10(i % 10) for i, t in enumerate(tag_order)}

    for ax, metric in zip(axes, metrics_to_plot):
        for xi, combo in enumerate(combos):
            sub = df[df["method_combo"] == combo]
            for _, row in sub.iterrows():
                x_jit = xi + (rng.random() - 0.5) * 0.25
                ax.scatter(x_jit, row[metric],
                           color=tag_to_color[row["dataset_tag"]],
                           s=40, alpha=0.8,
                           label=row["dataset_tag"])
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(range(len(combos)))
    axes[-1].set_xticklabels(combos, rotation=45, ha="right")

    handles_labels = {}
    for ax in axes:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            handles_labels[lbl] = h
    axes[0].legend(handles_labels.values(), handles_labels.keys(),
                   fontsize=8, loc="upper right", title="dataset_tag")

    fig.suptitle("Metric stability across dataset realisations", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "metric_stability.png", dpi=150)
    plt.close(fig)


def _colour_for(value: float, col: str, col_min: float, col_max: float):
    """Map a numeric cell to an RGBA colour.

    Uses the Reds→Greens spectrum so that higher-is-better columns render
    green at the top and red at the bottom; inverted for LOWER_IS_BETTER.
    """
    if pd.isna(value) or col_max == col_min:
        return (1.0, 1.0, 1.0, 1.0)
    norm = Normalize(vmin=col_min, vmax=col_max)
    t = norm(value)
    if col in LOWER_IS_BETTER:
        t = 1.0 - t
    return plt.cm.RdYlGn(t)


def save_metrics_table(all_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Dump the combined metrics table as CSV and as a rendered PNG.

    Numeric columns are colour-coded per column (green = best, red = worst,
    direction flipped for LOWER_IS_BETTER columns) and the per-column winner
    is bolded. The CSV is written unformatted for downstream processing.
    """
    csv_path = out_dir / "metrics_table.csv"
    all_metrics.to_csv(csv_path, index=False)

    df = all_metrics.copy()

    # Per-column min/max/best-row for colouring and winner bolding.
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    col_stats = {}
    for c in numeric_cols:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0:
            continue
        vmin, vmax = vals.min(), vals.max()
        best = vals.idxmin() if c in LOWER_IS_BETTER else vals.idxmax()
        col_stats[c] = (vmin, vmax, best)

    display = df.copy()
    for col in display.columns:
        if display[col].dtype.kind == "f":
            display[col] = display[col].map(
                lambda v: "" if pd.isna(v) else f"{v:.3f}"
            )

    n_rows, n_cols = display.shape
    fig_w = max(8, 1.1 * n_cols)
    fig_h = max(2, 0.35 * (n_rows + 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    col_index = {name: i for i, name in enumerate(display.columns)}
    for col, (vmin, vmax, best_idx) in col_stats.items():
        j = col_index[col]
        for i in range(n_rows):
            raw = df[col].iloc[i]
            cell = table[(i + 1, j)]  # +1 skips the header row
            cell.set_facecolor(_colour_for(raw, col, vmin, vmax))
            if i == best_idx:
                cell.get_text().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_dir / "metrics_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pivot_heatmaps(all_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Per-metric method_combo x dataset_tag heatmaps.

    Designed to make "which technique wins where" readable at a glance
    when a dataset axis is present. One subplot per metric. The best cell
    per column (per dataset_tag) is bolded so readers can trace a winner
    down each column. Skipped if there is only one dataset_tag.
    """
    if "dataset_tag" not in all_metrics.columns:
        return
    if all_metrics["dataset_tag"].nunique() < 2:
        print("  only one dataset_tag, skipping pivot heatmaps")
        return

    df = all_metrics[all_metrics["space"] == "embedding_2d"].copy()
    df["method_combo"] = df["run"].apply(
        lambda r: r.split("__", 1)[1] if "__" in r else r
    )

    metrics_to_plot = [
        ("ari", "ARI"),
        ("nmi", "NMI"),
        ("ami", "AMI"),
        ("classifier_accuracy", "Classifier accuracy"),
    ]
    available = [(c, l) for (c, l) in metrics_to_plot if c in df.columns]
    if not available:
        return

    combos = sorted(df["method_combo"].unique())
    tags = sorted(df["dataset_tag"].unique())

    ncols = 2 if len(available) > 1 else 1
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(6, 1.1 * len(tags) + 3) * ncols,
                 max(3, 0.5 * len(combos) + 1.5) * nrows),
        squeeze=False,
    )

    for idx, (col, label) in enumerate(available):
        ax = axes[idx // ncols][idx % ncols]
        pivot = df.pivot_table(
            index="method_combo", columns="dataset_tag", values=col,
            aggfunc="first",
        ).reindex(index=combos, columns=tags)

        arr = pivot.values.astype(float)
        invert = col in LOWER_IS_BETTER
        cmap = "RdYlGn_r" if invert else "RdYlGn"

        im = ax.imshow(arr, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(combos)))
        ax.set_yticklabels(combos, fontsize=8)
        ax.set_title(label, fontsize=10)

        # Annotate each cell; bold the per-column winner.
        col_best = (np.nanargmin(arr, axis=0) if invert
                    else np.nanargmax(arr, axis=0))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isnan(v):
                    continue
                weight = "bold" if i == col_best[j] else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, fontweight=weight, color="black")

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # Hide unused subplot slots.
    for k in range(len(available), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle("Per-metric winners across datasets", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "pivot_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out", type=str, default="figures/dashboard")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = SCRIPT_DIR / results_dir

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = SCRIPT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading runs from {results_dir}")
    all_metrics = load_runs(results_dir)
    n_runs = all_metrics["run"].nunique()
    print(f"Loaded {n_runs} runs ({len(all_metrics)} metric rows)")

    print("Writing metrics table")
    save_metrics_table(all_metrics, out_dir)

    print("Writing embedding grid")
    save_embedding_grid(results_dir, out_dir)

    print("Writing metric bars")
    save_metric_bars(all_metrics, out_dir)

    print("Writing classifier bars")
    save_classifier_bars(all_metrics, out_dir)

    print("Writing stability figure")
    save_stability_figure(all_metrics, out_dir)

    print("Writing pivot heatmaps")
    save_pivot_heatmaps(all_metrics, out_dir)

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
