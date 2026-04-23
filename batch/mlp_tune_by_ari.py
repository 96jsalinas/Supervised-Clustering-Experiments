"""MLP hyperparameter tuning sweep — LEGACY (ranks by ARI).

Kept for reproducibility of the 12 April 2026 tuning sweep. New tuning work
should use the in-pipeline `model.tune` block (see `pipeline/tuning.py` and
`configs/mlp_parity_tune.yaml`), which ranks by classifier metrics on a
held-out split via stratified K-fold CV.

Fixes SHAP + UMAP + HDBSCAN (same as mlp_baseline.yaml) and varies:
  - standardize  : whether to z-score inputs inside the MLP
  - hidden_sizes : network width/depth

Writes results to results/mlp_tune/<combo>/ and prints a ranked summary table.

Usage:
    python -m batch.mlp_tune_by_ari
    python -m batch.mlp_tune_by_ari --dry-run
    python -m batch.mlp_tune_by_ari --results-dir results/mlp_tune
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from data.synthetic import generate_data
from evaluation.figures import save_all_figures
from evaluation.metrics import compute_all_metrics
from pipeline.runner import PipelineRunner

_BASE_DATA = {
    "n_samples": 1000,
    "n_features": 50,
    "n_informative": 5,
    "n_clusters": 6,
    "cluster_std": 1.0,
    "center_box": [-2.0, 2.0],
    "random_state": 42,
}

_BASE_ATTR = {
    "method": "shap",
    "target_class": 1,
    "background_size": 100,
    "random_state": 42,
}

_BASE_RED = {
    "method": "umap",
    "random_state": 42,
    "params": {"n_components": 2, "n_neighbors": 200, "min_dist": 0.0},
}

_BASE_CLUST = {
    "method": "hdbscan",
    "params": {"min_cluster_size": 70},
}

_BASE_EVAL = {
    "external": ["ari", "nmi", "ami"],
    "internal": ["silhouette", "davies_bouldin", "calinski_harabasz"],
}

GRID = [
    (f"std{int(s)}_h{'x'.join(str(h) for h in arch)}", s, arch)
    for s in [False, True]
    for arch in [[64, 32], [128, 64], [256, 128]]
]


def build_config(standardize: bool, hidden_sizes: list) -> dict:
    return {
        "data": deepcopy(_BASE_DATA),
        "model": {
            "method": "mlp",
            "random_state": 42,
            "params": {
                "hidden_sizes": hidden_sizes,
                "dropout": 0.0,
                "epochs": 300,
                "lr": 0.001,
                "batch_size": 64,
                "weight_decay": 0.0,
                "val_fraction": 0.2,
                "patience": 30,
                "device": "cpu",
                "standardize": standardize,
            },
        },
        "attribution": deepcopy(_BASE_ATTR),
        "reduction": deepcopy(_BASE_RED),
        "clustering": deepcopy(_BASE_CLUST),
        "evaluation": deepcopy(_BASE_EVAL),
    }


def run_one(run_name: str, run_cfg: dict, results_root: Path) -> dict:
    output_dir = results_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    if figures_dir.exists():
        for stale in figures_dir.glob("*.png"):
            stale.unlink()

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)

    X, y_class, y_subcluster = generate_data(run_cfg["data"])
    runner = PipelineRunner(run_cfg)
    result = runner.run(X, y_class, y_subcluster)

    metrics_df = compute_all_metrics(result)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    np.savez(
        output_dir / "arrays.npz",
        embedding_2d=result.embedding_2d,
        cluster_labels_2d=result.cluster_labels_2d,
        cluster_labels_full=result.cluster_labels_full,
        y_subcluster=result.y_subcluster,
        y_class=result.y_class,
    )
    save_all_figures(result, figures_dir)

    row = metrics_df[metrics_df["space"] == "embedding_2d"].iloc[0]
    return {
        "combo": run_name,
        "ari": round(float(row["ari"]), 4),
        "nmi": round(float(row["nmi"]), 4),
        "ami": round(float(row["ami"]), 4),
        "n_clusters": int(row["n_clusters"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List combinations without running")
    parser.add_argument("--results-dir", default="results/mlp_tune",
                        help="Output root (default: results/mlp_tune)")
    args = parser.parse_args()

    print(f"MLP tuning grid: {len(GRID)} combinations\n")
    header = f"  {'combo':<30}  standardize  hidden_sizes"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tag, standardize, arch in GRID:
        print(f"  {tag:<30}  {str(standardize):<11}  {arch}")

    if args.dry_run:
        return

    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = SCRIPT_DIR / results_root
    results_root.mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    for i, (tag, standardize, hidden_sizes) in enumerate(GRID, start=1):
        print(f"\n=== [{i}/{len(GRID)}] {tag} ===")
        cfg = build_config(standardize, hidden_sizes)
        try:
            row = run_one(tag, cfg, results_root)
            rows.append(row)
            print(f"  ARI={row['ari']}  NMI={row['nmi']}  n_clusters={row['n_clusters']}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append((tag, str(exc)))

    if rows:
        print("\n=== Summary (ranked by ARI) ===")
        df = pd.DataFrame(rows).sort_values("ari", ascending=False)
        print(df.to_string(index=False))
        summary_path = results_root / "summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSummary written to {summary_path}")

    if failed:
        print("\nFailed runs:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
