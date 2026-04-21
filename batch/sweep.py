"""Run the Cartesian product of attribution x reduction x clustering methods
defined in a grid spec YAML. Each combination writes to
results/<attr>_<red>_<clust>/ with the same outputs as run_experiment.py.

Usage:
    python -m batch.sweep batch/full_grid.yaml
    python -m batch.sweep batch/full_grid.yaml --dry-run
"""

import argparse
import itertools
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from data.synthetic import generate_data
from pipeline.runner import PipelineRunner
from evaluation.metrics import compute_all_metrics
from evaluation.figures import save_all_figures


def _build_step_config(step_spec: dict, method: str) -> dict:
    """Merge `common` and per-method overrides for one pipeline step."""
    cfg = {"method": method}
    cfg.update(deepcopy(step_spec.get("common", {})))
    cfg.update(deepcopy(step_spec.get("per_method", {}).get(method, {})))
    return cfg


def build_configs(spec: dict) -> list[tuple[str, dict]]:
    """Enumerate all combinations from the grid spec.

    Returns a list of (run_name, run_config) tuples.
    """
    attrs = spec["attribution"]["methods"]
    reds = spec["reduction"]["methods"]
    clusts = spec["clustering"]["methods"]

    runs = []
    for a, r, c in itertools.product(attrs, reds, clusts):
        run_name = f"{a}_{r}_{c}"
        run_cfg = {
            "data": deepcopy(spec["data"]),
            "model": deepcopy(spec["model"]),
            "attribution": _build_step_config(spec["attribution"], a),
            "reduction": _build_step_config(spec["reduction"], r),
            "clustering": _build_step_config(spec["clustering"], c),
            "evaluation": deepcopy(spec.get("evaluation", {})),
        }
        runs.append((run_name, run_cfg))
    return runs


def run_one(run_name: str, run_cfg: dict, results_root: Path) -> None:
    output_dir = results_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)

    print(f"[{run_name}] generating data")
    X, y_class, y_subcluster = generate_data(run_cfg["data"])

    print(f"[{run_name}] running pipeline")
    runner = PipelineRunner(run_cfg)
    result = runner.run(X, y_class, y_subcluster)

    print(f"[{run_name}] computing metrics")
    metrics_df = compute_all_metrics(result)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    print(f"[{run_name}] saving figures")
    save_all_figures(result, figures_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=str, help="Path to grid spec YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate combinations without running them")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Where to write per-run output folders")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = SCRIPT_DIR / spec_path
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    runs = build_configs(spec)
    print(f"Total combinations: {len(runs)}")

    if args.dry_run:
        for name, _ in runs:
            print(f"  {name}")
        return

    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = SCRIPT_DIR / results_root
    results_root.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, (name, cfg) in enumerate(runs, start=1):
        print(f"\n=== [{i}/{len(runs)}] {name} ===")
        try:
            run_one(name, cfg, results_root)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {exc}")
            failed.append((name, str(exc)))

    if failed:
        print("\nFailed runs:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)
    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
