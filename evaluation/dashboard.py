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

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
