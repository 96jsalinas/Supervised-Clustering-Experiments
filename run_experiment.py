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

    data_cfg = config["data"]
    source = data_cfg.get("source", "synthetic")
    feature_names = None
    meta = None
    if source == "real":
        from data.real import load_real
        print("Loading real dataset...")
        X, y_class, y_subcluster, feature_names, meta = load_real(data_cfg)
        print(f"  Dataset: {meta['dataset']} "
              f"(n={meta['n_total']}, positives={meta['n_positive']}, "
              f"n_features={X.shape[1]})")
    elif source == "synthetic":
        print("Generating data...")
        X, y_class, y_subcluster = generate_data(data_cfg)
    else:
        raise ValueError(
            f"data.source must be 'synthetic' or 'real', got '{source}'."
        )

    print("Running pipeline...")
    runner = PipelineRunner(config)
    result = runner.run(X, y_class, y_subcluster)

    print("Computing metrics...")
    metrics_df = compute_all_metrics(result)

    # Cooper/Islam-style CV performance over the full dataset, reported as the
    # headline classifier number (comparable to the published baselines). Uses
    # the tuned params when a tuning pass ran. Gated by config so synthetic runs
    # do not pay for it.
    cv_cfg = (config.get("evaluation") or {}).get("cv_report") or {}
    if cv_cfg.get("enabled", False):
        from copy import deepcopy
        from pipeline.registry import MODELS
        from evaluation.classifier import cross_val_report
        model_cfg = deepcopy(config["model"])
        if result.tuning_selected is not None:
            model_cfg.setdefault("params", {}).update(
                result.tuning_selected["params"]
            )
        print("Computing cross-validated classifier performance...")
        cv_report = cross_val_report(
            MODELS[model_cfg["method"]],
            model_cfg,
            X,
            y_class,
            n_splits=int(cv_cfg.get("n_splits", 10)),
            random_state=int(cv_cfg.get("random_state", 42)),
        )
        for k, v in cv_report.items():
            metrics_df[k] = v
        acc = cv_report["cv_accuracy_mean"]
        auc = cv_report.get("cv_auc_mean")
        auc_str = f", AUC={auc:.4f}" if auc is not None else ""
        print(f"  {cv_report['cv_n_splits']}-fold CV: "
              f"accuracy={acc:.4f}{auc_str}")

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

    # Persist HDBSCAN grid-search artefacts when auto_select ran, so the
    # silhouette curve and the chosen parameters are recoverable.
    cmeta = result.clustering_meta or {}
    for tag in ("2d", "full"):
        elbow_df = cmeta.get(f"elbow_df_{tag}")
        if elbow_df is not None:
            elbow_df.to_csv(output_dir / f"hdbscan_grid_{tag}.csv", index=False)
        sel_params = cmeta.get(f"selected_params_{tag}")
        if sel_params is not None:
            print(f"  HDBSCAN selected ({tag}): {sel_params} "
                  f"-> k={cmeta.get(f'selected_k_{tag}')}")

    print("Saving arrays...")
    import numpy as np
    # y_subcluster is None on real data; only persist it when present so the
    # archive stays a clean numeric array rather than a 0-d object array.
    array_kwargs = dict(
        embedding_2d=result.embedding_2d,
        cluster_labels_2d=result.cluster_labels_2d,
        cluster_labels_full=result.cluster_labels_full,
        y_class=result.y_class,
        subset_mask=result._mask(),
    )
    if result.y_subcluster is not None:
        array_kwargs["y_subcluster"] = result.y_subcluster
    np.savez(output_dir / "arrays.npz", **array_kwargs)

    print("Saving figures...")
    save_all_figures(result, figures_dir, config=config,
                     feature_names=feature_names)

    # Cooper's validation arm: cluster the raw features (no SHAP) with the same
    # reduction + clustering and contrast the silhouette. Gated by config.
    raw_cfg = (config.get("evaluation") or {}).get("raw_baseline") or {}
    if raw_cfg.get("enabled", False):
        from evaluation.raw_baseline import compute_raw_baseline
        from evaluation.figures import save_shap_vs_raw
        from evaluation.metrics import _compute_internal
        print("Computing raw-feature baseline (no SHAP)...")
        X_sub_raw = X[result._mask()]
        raw = compute_raw_baseline(X_sub_raw, config)
        sil_shap = _compute_internal(
            result.embedding_2d, result.cluster_labels_2d
        )["silhouette"]
        metrics_df["raw_silhouette"] = raw["silhouette"]
        metrics_df["raw_n_clusters"] = raw["n_clusters"]
        metrics_df["raw_n_noise"] = raw["n_noise"]
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        save_shap_vs_raw(result, raw, sil_shap, figures_dir)
        print(f"  SHAP silhouette={sil_shap:.4f} vs "
              f"raw silhouette={raw['silhouette']:.4f} "
              f"(raw k={raw['n_clusters']}, noise={raw['n_noise']})")

    # Qualitative cluster description: per-cluster symptom prevalence plus a
    # shallow multiclass decision tree (figure + rules), in original units.
    # Needs the loader's inverse-transform metadata, so it is real-data only.
    desc_cfg = (config.get("evaluation") or {}).get("cluster_description") or {}
    if desc_cfg.get("enabled", False):
        if meta is None or feature_names is None:
            print("  Skipping cluster description: needs real-data metadata.")
        else:
            from evaluation.cluster_description import describe_clusters
            # tree_max_depth: max comparison terms per rule (Cooper uses 2).
            md = int(desc_cfg.get("tree_max_depth", 2))
            print("Describing clusters (prevalence + one-vs-all rules)...")
            desc = describe_clusters(
                result, feature_names, meta, output_dir, figures_dir,
                max_depth=md,
            )
            agr = desc["agreement"]
            print(f"  2D labeling: {agr['n_clusters_2d']} clusters; "
                  f"full-space: {agr['n_clusters_full']} clusters; "
                  f"ARI(2D,full)={agr['ari_2d_vs_full']:.3f}")
            if (agr["ari_2d_vs_full"] < 0.5
                    or agr["n_clusters_2d"] != agr["n_clusters_full"]):
                print("  NOTE: 2D and full-space clusterings diverge "
                      "noticeably; full-space tables saved alongside for review.")
            n_rules = len(desc["rules"]["2d"])
            print(f"  Wrote prevalence tables and {n_rules} 2D tree rules.")

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
