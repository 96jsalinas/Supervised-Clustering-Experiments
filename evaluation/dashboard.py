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

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_runs(results_dir: Path) -> pd.DataFrame:
    """Concatenate every run's metrics.csv with a leading `run` column."""
    frames = []
    for run_dir in sorted(results_dir.iterdir()):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.is_file():
            continue
        df = pd.read_csv(metrics_path)
        df.insert(0, "run", run_dir.name)
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


def save_metrics_table(all_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Dump the combined metrics table as CSV and as a rendered PNG."""
    csv_path = out_dir / "metrics_table.csv"
    all_metrics.to_csv(csv_path, index=False)

    display = all_metrics.copy()
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
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_table.png", dpi=150, bbox_inches="tight")
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

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
