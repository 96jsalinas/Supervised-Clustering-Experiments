"""Entry point for running a supervised clustering experiment.

Usage:
    python run_experiment.py configs/cooper_dbscan.yaml
"""

import shutil
import sys
from pathlib import Path

import yaml

# Resolve paths relative to this script so it works from any CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data.synthetic import generate_data
from pipeline.runner import PipelineRunner
from evaluation.metrics import compute_all_metrics
from evaluation.figures import save_all_figures


def main(config_path: str):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_name = config_path.stem
    output_dir = SCRIPT_DIR / "results" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    # Copy config verbatim for reproducibility
    shutil.copy(config_path, output_dir / "config.yaml")

    print(f"Running experiment: {run_name}")
    print(f"Output directory:   {output_dir}")

    print("Generating data...")
    X, y_class, y_subcluster = generate_data(config["data"])

    print("Running pipeline...")
    runner = PipelineRunner(config)
    result = runner.run(X, y_class, y_subcluster)

    print("Computing metrics...")
    metrics_df = compute_all_metrics(result)

    if result.tuning_selected is not None:
        sel = result.tuning_selected
        metrics_df["tuned"] = True
        metrics_df["cv_score"] = sel["cv_score_mean"]
        metrics_df["cv_score_std"] = sel["cv_score_std"]
        metrics_df["cv_scoring"] = sel["scoring"]
        for k, v in sel["params"].items():
            metrics_df[f"selected_{k}"] = (
                str(v) if isinstance(v, (list, tuple)) else v
            )
        result.tuning_grid.to_csv(
            output_dir / "tuning_grid.csv", index=False
        )
        with open(output_dir / "tuning_selected.yaml", "w") as f:
            yaml.safe_dump(sel, f, sort_keys=False)
        print(f"Tuning winner: {sel['combo_tag']} "
              f"({sel['scoring']}={sel['cv_score_mean']:.4f})")
    else:
        metrics_df["tuned"] = False

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    print("Saving arrays...")
    import numpy as np
    np.savez(
        output_dir / "arrays.npz",
        embedding_2d=result.embedding_2d,
        cluster_labels_2d=result.cluster_labels_2d,
        cluster_labels_full=result.cluster_labels_full,
        y_subcluster=result.y_subcluster,
        y_class=result.y_class,
    )

    print("Saving figures...")
    save_all_figures(result, figures_dir)

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
