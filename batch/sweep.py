"""Run the Cartesian product of (dataset x model x attribution x reduction x
clustering) combinations defined in a grid spec YAML. Each combination writes
to results/<dataset_tag>__<model_tag>__<attr>_<red>_<clust>/ with the same
outputs as run_experiment.py.

Classifier tuning, when enabled, is hoisted to the (dataset x model) level so
a single winner propagates to every (attr, red, clust) combo for that cell.

Usage:
    python -m batch.sweep batch/full_grid.yaml
    python -m batch.sweep batch/full_grid.yaml --dry-run
"""

import argparse
import itertools
import os
import sys
import warnings
from copy import deepcopy
from pathlib import Path

# Silence three benign warnings that completely flood the sweep log
# (~9.9k warning lines on the 24-Apr run, vs ~5.8k actual progress lines):
#   - LightGBM raises "X does not have valid feature names" on every
#     predict because we feed it numpy arrays after fitting with a DataFrame.
#   - UMAP prints "n_jobs value 1 overridden" once per fit when
#     random_state is set (we always set it for reproducibility).
#   - sklearn's KMeans warns about an MKL memory leak on Windows when
#     n_chunks < n_threads. Setting OMP_NUM_THREADS=4 before sklearn is
#     imported is the recommended fix and removes the warning at source.
# Filters are message-scoped, not blanket "ignore", so any new or
# unrelated warning still surfaces.
os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value .* overridden to 1 by setting random_state",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"KMeans is known to have a memory leak on Windows with MKL",
    category=UserWarning,
)

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


def _resolve_models(spec: dict) -> list[dict]:
    """Return the list of model specs to iterate.

    If the top-level `models:` key is present, use it verbatim. Otherwise fall
    back to `model:` (singular) and synthesise a tag from the method name, so
    existing single-model specs continue to work without edits.
    """
    if "models" in spec and spec["models"]:
        models = []
        for i, m in enumerate(spec["models"]):
            m = deepcopy(m)
            m.setdefault("tag", m.get("method", f"model{i}"))
            models.append(m)
        return models
    single = deepcopy(spec.get("model", {}) or {})
    single.setdefault("tag", single.get("method", "model"))
    return [single]


def build_configs(spec: dict) -> list[tuple[str, dict, str, str]]:
    """Enumerate all combinations from the grid spec.

    If `datasets:` is present it is a list of `{tag, override}` entries
    merged into the baseline `data:` block. If `models:` is present it is a
    list of model configs each with a `tag` field. The Cartesian product is
    datasets × models × attribution × reduction × clustering.

    Returns a list of (run_name, run_config, dataset_tag, model_tag) tuples.
    """
    attrs = spec["attribution"]["methods"]
    reds = spec["reduction"]["methods"]
    clusts = spec["clustering"]["methods"]

    datasets = spec.get("datasets")
    if datasets is None:
        datasets = [{"tag": None, "override": {}}]

    models = _resolve_models(spec)

    runs = []
    for ds in datasets:
        ds_tag = ds.get("tag")
        data_cfg = deepcopy(spec["data"])
        data_cfg.update(ds.get("override", {}))
        for m_cfg in models:
            model_tag = m_cfg["tag"]
            model_block = {k: v for k, v in m_cfg.items() if k != "tag"}
            for a, r, c in itertools.product(attrs, reds, clusts):
                method_name = f"{a}_{r}_{c}"
                parts = [p for p in (ds_tag, model_tag) if p]
                prefix = "__".join(parts)
                run_name = (
                    f"{prefix}__{method_name}" if prefix else method_name
                )
                run_cfg = {
                    "data": data_cfg,
                    "model": deepcopy(model_block),
                    "attribution": _build_step_config(spec["attribution"], a),
                    "reduction": _build_step_config(spec["reduction"], r),
                    "clustering": _build_step_config(spec["clustering"], c),
                    "evaluation": deepcopy(spec.get("evaluation", {})),
                }
                runs.append(
                    (run_name, run_cfg, ds_tag or "default", model_tag)
                )
    return runs


def run_one(run_name: str, run_cfg: dict, results_root: Path,
            dataset_tag: str = "default",
            model_tag: str = "model",
            tuning_info: dict | None = None) -> None:
    """Run a single (attr × red × clust) combination for one (dataset, model).

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
    metrics_df.insert(0, "model_tag", model_tag)
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

    cmeta = result.clustering_meta or {}
    for suffix in ("2d", "full"):
        elbow_df = cmeta.get(f"elbow_df_{suffix}")
        if elbow_df is not None:
            elbow_df.to_csv(output_dir / f"elbow_curve_{suffix}.csv", index=False)

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
        for name, _, ds_tag, m_tag in runs:
            print(f"  [{ds_tag} | {m_tag}] {name}")
        return

    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = SCRIPT_DIR / results_root
    results_root.mkdir(parents=True, exist_ok=True)

    # Group combos by (dataset_tag, model_tag). Tuning is hoisted to this
    # level — it depends on the model and the data, not on the downstream
    # (attr, red, clust) choices, so running it per combo would be both
    # wasteful and noisy (different CV folds per combo).
    by_cell: dict[tuple[str, str], list] = {}
    for entry in runs:
        by_cell.setdefault((entry[2], entry[3]), []).append(entry)

    tuning_by_cell: dict[tuple[str, str], dict] = {}
    eval_cfg = spec.get("evaluation", {}) or {}
    test_fraction = float(eval_cfg.get("test_fraction", 0.2))

    # Cache generated data per dataset_tag so we don't regenerate per model.
    data_cache: dict[str, tuple] = {}

    for (ds_tag, m_tag), entries in by_cell.items():
        sample_cfg = entries[0][1]
        model_cfg = sample_cfg["model"]
        tune_cfg = (model_cfg or {}).get("tune") or {}
        if not tune_cfg.get("enabled", False):
            continue

        if ds_tag not in data_cache:
            X, y_class, _ = generate_data(sample_cfg["data"])
            data_cache[ds_tag] = (X, y_class)
        else:
            X, y_class = data_cache[ds_tag]

        split_seed = int(eval_cfg.get(
            "split_seed", model_cfg.get("random_state", 42)
        ))
        train_idx, _ = train_test_split(
            np.arange(len(X)),
            test_size=test_fraction,
            stratify=y_class,
            random_state=split_seed,
        )
        print(f"\n=== Tuning classifier for [{ds_tag} | {m_tag}] ===")
        selected, grid_rows = tune_classifier(
            model_cls=MODELS[model_cfg["method"]],
            base_config=model_cfg,
            tune_cfg=tune_cfg,
            X_train=X[train_idx],
            y_train=y_class[train_idx],
        )
        tune_dir = results_root / f"{ds_tag}__{m_tag}__tuning"
        tune_dir.mkdir(parents=True, exist_ok=True)
        grid_rows.to_csv(tune_dir / "tuning_grid.csv", index=False)
        with open(tune_dir / "tuning_selected.yaml", "w") as f:
            yaml.safe_dump(selected, f, sort_keys=False)
        print(f"  winner: {selected['combo_tag']} "
              f"{selected['scoring']}={selected['cv_score_mean']:.4f}")
        tuning_by_cell[(ds_tag, m_tag)] = selected

    failed = []
    total = sum(len(v) for v in by_cell.values())
    idx = 0
    for (ds_tag, m_tag), entries in by_cell.items():
        selected = tuning_by_cell.get((ds_tag, m_tag))
        for (name, cfg, _ds, _m) in entries:
            idx += 1
            print(f"\n=== [{idx}/{total}] {name} ===")
            combo_cfg = cfg
            if selected is not None:
                combo_cfg = deepcopy(cfg)
                combo_cfg["model"].setdefault("params", {}).update(
                    selected["params"]
                )
                combo_cfg["model"].setdefault("tune", {})["enabled"] = False
            try:
                run_one(name, combo_cfg, results_root,
                        dataset_tag=ds_tag, model_tag=m_tag,
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
