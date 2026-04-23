# Handoff — overnight full-comparison sweep

**Status:** disposable. This file is a bridge between two Claude Code sessions. Delete once the overnight sweep is done and any follow-up analysis is complete.

**Audience:** a fresh Claude Code session with none of today's conversation context, picking up tomorrow.

---

## TL;DR

Run:

```
rm -rf results_smoke_fullcomparison figures/dashboard_smoke_fullcomparison
python -m batch.sweep batch/full_comparison_grid.yaml \
    --results-dir results_full_comparison 2>&1 \
    | tee -a full_comparison_log.txt
```

Sleep. Wake up. Run:

```
python -m evaluation.dashboard \
    --results-dir results_full_comparison \
    --out figures/dashboard_full_comparison
```

Look at `figures/dashboard_full_comparison/pivot_heatmaps.png` — that one artefact is what answers Pablo's questions.

---

## Why this sweep exists

Running log: `../Supervised Clustering/meetings/21_04_2026_followup_findings.md` has the full timeline. Short version:

1. **Pablo's question (21 Apr):** is LightGBM genuinely a better classifier than the MLP on our synthetic data, or is the MLP under-tuned?
2. **First answer (23 Apr, via a 72-combo in-pipeline tuning sweep on seed=42):** MLP ceiling is ~0.79 accuracy; LightGBM 0.87. Parity **not** achievable with the reachable hyperparameter grid.
3. **Open concern:** the 72-combo sweep ran on a single generator family (`make_blobs`, axis-aligned Gaussians — tree-friendly by construction) and a single seed. Trends may be seed-dependent or generator-dependent. Pablo needs evidence across both axes.

The overnight sweep addresses (3) systematically:

- **Data parameters, one-axis-at-a-time:** `n_informative ∈ {3, 5, 10}`, `n_clusters ∈ {4, 6, 8}`, `cluster_std ∈ {0.7, 1.0, 1.3}`, baseline shared. 7 distinct parameter combos.
- **Generator families:** `blobs` (unchanged) and `rotated` (post-hoc random orthogonal rotation of the informative subspace — see `README.md` "Optional orthogonal rotation" for Pablo-readable prose).
- **Seeds:** 3 replicates per distribution (`s1`, `s2`, `s3`), for within-cell sampling variance.
- **Models:** MLP, LightGBM — each with its own per-(dataset × model) tuning pass.
- **Method combos:** SHAP × {UMAP, PCA} × {HDBSCAN, k-means} = 4 per cell.

Total: 42 dataset cells × 2 models × 4 combos = **336 pipeline runs** + **84 tuning sweeps**.

## Expected runtime

~5.5 h on CPU. Breakdown in the plan file (`C:\Users\96jsa\.claude\plans\let-s-plan-how-to-cuddly-wirth.md`). If the machine is idle it will finish comfortably overnight.

## What to look at tomorrow

Run the dashboard first (command above). Then:

### `figures/dashboard_full_comparison/pivot_heatmaps.png` — the main artefact

Rows = metrics (ARI, NMI, AMI, classifier_accuracy). Columns = models (MLP, LightGBM). Each cell = method-combo × dataset-tag, colour-scaled green (best) → red (worst), with the per-column winner bolded. The key comparisons:

- **Classifier_accuracy row:** does the MLP-vs-LightGBM gap close on any cell, or does LightGBM dominate everywhere? Pay attention to whether the `rotated_*` cells behave differently from `blobs_*` cells — that tells you whether the gap is an axis-alignment artefact.
- **ARI row:** does the subgroup-recovery gap track the classifier_accuracy gap, or is there an independent effect? If MLP+LightGBM reach parity in classifier accuracy on some cells but LightGBM still wins ARI there, the attribution-surface argument holds.
- **Within each heatmap, per-dataset winners:** the bolded cells tell you which method combo wins each data realisation. Trends across seed replicates (columns `*_s1`, `*_s2`, `*_s3` at the same `{gen}_{axis}_{level}`) indicate whether a finding is stable.

### `figures/dashboard_full_comparison/metric_stability.png`

Strip plot per (metric, model). x-axis = method combo. Each point = one dataset. Spread within a combo = how much the metric varies across data realisations. Tight clusters = robust method. Big spreads = seed-dependent.

### `figures/dashboard_full_comparison/metrics_table.png`

Colour-coded full metrics table. Best value in each column is bolded. Useful for sanity checks and finding outliers.

### Tuning artefacts

Every `<dataset_tag>__<model_tag>__tuning/` directory holds:

- `tuning_selected.yaml` — the winning hyperparameters and CV score.
- `tuning_grid.csv` — every (grid point × fold) CV score.

If you want to see whether the tuning found meaningfully different winners on different datasets (and especially on different generators), scan the `tuning_selected.yaml` files — a Python one-liner in the dashboard directory can aggregate them if needed.

## What Pablo needs out of this (for the email)

The followup-findings file (`../Supervised Clustering/meetings/21_04_2026_followup_findings.md`) has the running narrative. After reading the sweep results, a fresh session should **append** a new section to that file titled `24 April follow-up — full comparison sweep` containing:

- One headline sentence: does rotation close the MLP-vs-LightGBM classifier gap? (yes / no / partially).
- Does the answer change across parameter axes (n_informative / n_clusters / cluster_std)?
- Does the answer change across seeds, or is the trend stable?
- For the subgroup recovery metrics: does LightGBM still dominate ARI/NMI/AMI even when the classifier gap is closed?
- The headline pivot heatmap figure path for the email attachment.

Leave the headline bullets near the top of that file intact (they're what gets pasted into the email); append the new section below with full detail.

## If something goes wrong overnight

The sweep writes every run to its own directory as it goes. If it crashes partway, the partial results still work for the dashboard — just run the dashboard on whatever is in `results_full_comparison/` and note which cells are missing.

Check `full_comparison_log.txt` for the failure point. Each run is self-contained, so re-running individual cells by copy-editing a small YAML and invoking `run_experiment.py` is cheap.

## Infrastructure cheat-sheet (for the fresh session)

- `batch/sweep.py` accepts `models: [...]` (list) alongside / instead of the single `model:` block. Per-(dataset, model) tuning is hoisted out of the combo loop. Run names are `<dataset_tag>__<model_tag>__<attr>_<red>_<clust>`; `metrics.csv` carries both `dataset_tag` and `model_tag` columns.
- `pipeline/tuning.py` provides `tune_classifier(model_cls, base_config, tune_cfg, X_train, y_train) -> (selected, grid_rows)`. Scoring options: `accuracy | auc | f1_macro | log_loss`.
- `data/synthetic.py` has `rotate_informative: bool` (default `False`) — applies a random orthogonal rotation to the informative subspace of `make_blobs`'s output. Cluster identities, per-cluster counts, and all pairwise centroid distances are preserved exactly. Comment block at the rotation site and a prose note in `README.md` explain what is and isn't preserved.
- `evaluation/dashboard.py` pivot heatmaps and stability figure are now model-aware (column facet per model_tag).

## Files to leave alone

- The 72-combo parity tune output (`results/mlp_parity_tune/`) — keep for reference; it's what answered Pablo's first question.
- `batch/mlp_tune_by_ari.py` — legacy ARI-ranked MLP tuner, kept for 12 Apr reproducibility.
- `batch/robustness_grid.yaml` — prior (24-run, single-model) sweep; kept as historical.

## Delete this file when done

After the overnight run, the dashboard is rendered, and the followup-findings file has been updated with the new section, delete `HANDOFF_OVERNIGHT_SWEEP.md` (this file) from the repo.
