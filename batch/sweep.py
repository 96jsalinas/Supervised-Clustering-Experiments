"""Run the Cartesian product of attribution x reduction x clustering methods
defined in a grid spec YAML. Each combination writes to
results/<attr>_<red>_<clust>/ with the same outputs as run_experiment.py.

Usage:
    python -m batch.sweep batch/full_grid.yaml
    python -m batch.sweep batch/full_grid.yaml --dry-run
"""

import argparse
import itertools
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from sklearn.model_selection import train_test_split

from data.synthetic import generate_data
from pipeline.runner import PipelineRunner
from pipeline.registry import MODELS
from pipeline.tuning import tune_classifier
from evaluation.metrics import compute_all_metrics
from evaluation.figures import save_all_figures


def _build_step_config(step_spec: dict, method: str) -> dict:
    """Merge `common` and per-method overrides for one pipeline step.

    The merge is shallow: if both `common` and `per_method[method]`
    define the same key (e.g. `params:`), the per-method value
    replaces the common value wholesale.
    """
    cfg = {"method": method}
    cfg.update(deepcopy(step_spec.get("common", {})))
    cfg.update(deepcopy(step_spec.get("per_method", {}).get(method, {})))
    return cfg


def build_configs(spec: dict) -> list[tuple[str, dict]]:
    """Enumerate all combinations from the grid spec.

    If `datasets:` is present in the spec it must be a list of dicts with a
    `tag:` key (short human-readable name) and an `override:` key (mapping
    merged into the baseline `data:` block). The Cartesian product is then
    taken over datasets x attribution x reduction x clustering and each run
    directory is named `<dataset_tag>__<attr>_<red>_<clust>`.

    Returns a list of (run_name, run_config, dataset_tag) tuples.
    """
    attrs = spec["attribution"]["methods"]
    reds = spec["reduction"]["methods"]
    clusts = spec["clustering"]["methods"]

    datasets = spec.get("datasets")
    if datasets is None:
        datasets = [{"tag": None, "override": {}}]

    runs = []
    for ds in datasets:
        tag = ds.get("tag")
        data_cfg = deepcopy(spec["data"])
        data_cfg.update(ds.get("override", {}))
        for a, r, c in itertools.product(attrs, reds, clusts):
            method_name = f"{a}_{r}_{c}"
            run_name = f"{tag}__{method_name}" if tag else method_name
            run_cfg = {
                "data": data_cfg,
                "model": deepcopy(spec["model"]),
                "attribution": _build_step_config(spec["attribution"], a),
                "reduction": _build_step_config(spec["reduction"], r),
                "clustering": _build_step_config(spec["clustering"], c),
                "evaluation": deepcopy(spec.get("evaluation", {})),
            }
            runs.append((run_name, run_cfg, tag or "default"))
    return runs


def run_one(run_name: str, run_cfg: dict, results_root: Path,
            dataset_tag: str = "default",
            tuning_info: dict | None = None) -> None:
    """Run a single (attr × red × clust) combination.

    If `tuning_info` is provided, the runner-level tuning pass is skipped
    (the combo's `model.tune.enabled` is forced to False) and the
    already-selected hyperparameters are annotated onto `metrics.csv`.
    """
    output_dir = results_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    # Clean stale PNGs so a rename upstream does not leave behind
    # obsolete figures from a previous run.
    if figures_dir.exists():
        for stale in figures_dir.glob("*.png"):
            stale.unlink()

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)

    print(f"[{run_name}] generating data")
    X, y_class, y_subcluster = generate_data(run_cfg["data"])

    print(f"[{run_name}] running pipeline")
    runner = PipelineRunner(run_cfg)
    result = runner.run(X, y_class, y_subcluster)

    print(f"[{run_name}] computing metrics")
    metrics_df = compute_all_metrics(result)
    metrics_df.insert(0, "dataset_tag", dataset_tag)
    if tuning_info is not None:
        metrics_df["tuned"] = True
        metrics_df["cv_score"] = tuning_info["cv_score_mean"]
        metrics_df["cv_score_std"] = tuning_info["cv_score_std"]
        metrics_df["cv_scoring"] = tuning_info["scoring"]
        for k, v in tuning_info["params"].items():
            metrics_df[f"selected_{k}"] = (
                str(v) if isinstance(v, (list, tuple)) else v
            )
    else:
        metrics_df["tuned"] = False
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    np.savez(
        output_dir / "arrays.npz",
        embedding_2d=result.embedding_2d,
        cluster_labels_2d=result.cluster_labels_2d,
        cluster_labels_full=result.cluster_labels_full,
        y_subcluster=result.y_subcluster,
        y_class=result.y_class,
    )

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
        for name, _, tag in runs:
            print(f"  [{tag}] {name}")
        return

    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = SCRIPT_DIR / results_root
    results_root.mkdir(parents=True, exist_ok=True)

    # Group combos by dataset_tag so tuning can be hoisted out of the inner
    # loop — tuning is model-level, not method-level, and running it per
    # combo would be both wasteful and noisy (different CV folds per combo).
    by_dataset: dict[str, list] = {}
    for entry in runs:
        by_dataset.setdefault(entry[2], []).append(entry)

    tuning_by_dataset: dict[str, dict] = {}
    model_spec = spec.get("model", {}) or {}
    tune_cfg = model_spec.get("tune") or {}
    if tune_cfg.get("enabled", False):
        eval_cfg = spec.get("evaluation", {}) or {}
        test_fraction = float(eval_cfg.get("test_fraction", 0.2))
        split_seed = int(eval_cfg.get(
            "split_seed", model_spec.get("random_state", 42)
        ))
        for tag, entries in by_dataset.items():
            sample_cfg = entries[0][1]  # all combos in a dataset share data
            print(f"\n=== Tuning classifier for dataset '{tag}' ===")
            X, y_class, _ = generate_data(sample_cfg["data"])
            train_idx, _ = train_test_split(
                np.arange(len(X)),
                test_size=test_fraction,
                stratify=y_class,
                random_state=split_seed,
            )
            selected, grid_rows = tune_classifier(
                model_cls=MODELS[model_spec["method"]],
                base_config=model_spec,
                tune_cfg=tune_cfg,
                X_train=X[train_idx],
                y_train=y_class[train_idx],
            )
            tune_dir = results_root / f"{tag}__tuning"
            tune_dir.mkdir(parents=True, exist_ok=True)
            grid_rows.to_csv(tune_dir / "tuning_grid.csv", index=False)
            with open(tune_dir / "tuning_selected.yaml", "w") as f:
                yaml.safe_dump(selected, f, sort_keys=False)
            print(f"  winner: {selected['combo_tag']} "
                  f"{selected['scoring']}={selected['cv_score_mean']:.4f}")
            tuning_by_dataset[tag] = selected

    failed = []
    total = sum(len(v) for v in by_dataset.values())
    idx = 0
    for tag, entries in by_dataset.items():
        selected = tuning_by_dataset.get(tag)
        for (name, cfg, _tag) in entries:
            idx += 1
            print(f"\n=== [{idx}/{total}] {name} ===")
            combo_cfg = cfg
            if selected is not None:
                # Inject tuned params, then disable runner-level tuning so the
                # combo does not re-tune; the winner is shared across combos.
                combo_cfg = deepcopy(cfg)
                combo_cfg["model"].setdefault("params", {}).update(
                    selected["params"]
                )
                combo_cfg["model"].setdefault("tune", {})["enabled"] = False
            try:
                run_one(name, combo_cfg, results_root, dataset_tag=tag,
                        tuning_info=selected)
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
