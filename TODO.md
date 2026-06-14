# TODO

Outstanding work for the experimental framework. Items are grouped by category.

## Pipeline methods to implement

- [x] **MLP model** (`pipeline/models/mlp.py`) -- PyTorch Linear+ReLU stack, the thesis baseline.
- [x] **LRP attributor** (`pipeline/attribution/lrp.py`) -- Layer-wise Relevance Propagation via zennit.
- [x] **LIME attributor** (`pipeline/attribution/lime.py`).
- [x] **PCA reducer** (`pipeline/reduction/pca_reducer.py`).
- [x] **t-SNE reducer** (`pipeline/reduction/tsne_reducer.py`).
- [x] **PaCMAP reducer** (`pipeline/reduction/pacmap_reducer.py`).
- [x] **k-means clusterer** (`pipeline/clustering/kmeans_clusterer.py`).

## Evaluation and analysis

- [x] **UMAP of SHAP values colored by subcluster** (`save_umap_shap_subcluster_labels`).
- [x] **Per-cluster SHAP profile plots** (`save_per_cluster_shap_profile`, small-multiple bar grid sorted by global importance).
- [x] **Clusters in full attribution space vs 2D** side-by-side (`save_clusters_no_dr_vs_dr`).
- [x] **Timing instrumentation** -- per-step wall-clock time recorded in `metrics.csv`.
- [x] **Cross-run dashboard** (`evaluation/dashboard.py`) -- aggregated metrics table, embedding grid, metric bars, classifier bars, stability figure, colour-coded pivot heatmaps.
- [x] **Classifier-level metrics** -- stratified train/test split inside `PipelineRunner`; accuracy/AUC/F1/log_loss emitted to `metrics.csv` per run.
- [x] **In-pipeline classifier tuning** -- optional `model.tune` block runs stratified K-fold CV over a hyperparameter grid before the rest of the pipeline (`pipeline/tuning.py`, `evaluation/classifier.py`). Model-agnostic.
- [~] **MLP hyperparameter tuning** -- 72-combo parity sweep run (activation × label_smoothing × lr × dropout × hidden_sizes). MLP ceiling on baseline data is ~0.79 accuracy vs LightGBM's 0.87. Parity not achievable; treated as a thesis finding rather than an outstanding task.
- [x] **Pair-counting F-measure** -- Larsen & Aone (1999) variant implemented in `evaluation/metrics.py` via contingency matrix. New `f_measure` column in `metrics.csv`. Reference: Taha & Hanbury (2015), BMC Medical Imaging.
- [x] **F-measure and attribution timing figures** -- `analyze_sweep.R` extended with F-measure boxplot panels (mirroring ARI set) and LRP-vs-SHAP timing + cost-benefit figures (self-activate when >1 attribution method in the CSV).
- [x] **`informative_fraction` parametrisation** -- `data/synthetic.py` accepts `informative_fraction` (float) as an alternative to `n_informative` (int); errors out if both are set. All sweep YAMLs updated to fraction-based grid `{0.1, 0.2, 0.4}` with 0.2 as baseline.
- [x] **ARI ground-truth label verified** -- confirmed `y_subcluster` (not `y_class`) is passed to all three external metrics in `evaluation/metrics.py:86`. No code change needed.
- **Dropped:** DBSCAN epsilon tuning -- DBSCAN was removed from the default grid (fixed `eps` is not a fair comparison vs HDBSCAN's auto-selection).

## Infrastructure

- [x] **Batch runner** (`batch/sweep.py`) -- Cartesian-product sweep with `--dry-run`, per-run result folders, shallow method-override merge, optional `datasets:` axis, optional `models:` axis, per-(dataset × model) classifier tuning hoisted out of the combo loop.
- [x] **`requirements.txt`** -- pinned dependency set generated from the working environment (see `requirements.txt` and `SETUP.md`).
- [ ] **Logging** -- replace `print()` statements with Python `logging` for configurable verbosity.

## Experiment design

- [x] **Multiple synthetic datasets** -- `datasets:` axis in grid specs; `batch/robustness_grid.yaml` covers seed replicates + easy/medium/hard scenarios.
- [x] **Stability analysis** -- 24-run robustness sweep across 3 seeds and 3 difficulty scenarios; stability figure + pivot heatmaps in the dashboard.
- [x] **Second generator family (axis-alignment test)** -- `rotate_informative: true` applies a random orthogonal rotation to `make_blobs`'s informative subspace; preserves cluster identities and pairwise centroid distances exactly, breaks tree-friendly axis-alignment.
- [x] **Full comparison sweep** -- `batch/full_comparison_grid.yaml` (42 dataset cells × 2 models × 4 method combos = 336 runs) executed overnight 2026-04-23/24. Outputs in `results_full_comparison/` and `figures/dashboard_full_comparison/`. Findings drove the design of the two follow-up sweeps below.
- [x] **Sweep A — data variety** -- `batch/sweep_a_data_variety.yaml` (74 dataset cells × 2 models × 4 method combos = 592 runs). Adds `n_features` and `center_box` (overlap) as data axes, drops the `rotated` family, concentrates seeds on the high-variance `std_*` cells. Run; outputs in `results_sweep_a/`.
- [x] **Sweep B — LRP vs SHAP** -- `batch/sweep_b_lrp_vs_shap.yaml` + companion `batch/sweep_b_lgbm_control.yaml` (31 dataset cells, MLP with SHAP+LRP plus LightGBM-SHAP control on the same cells, 372 runs total). Run; outputs in `results_sweep_b/`.
- [x] **Per-dataset clustering params** -- replaced fixed `k=6` with kneedle elbow auto-selection (`auto_select: {enabled: true, k_min: 2, k_max: 15}`) in `pipeline/clustering/kmeans_clusterer.py`. `selected_k` written to `metrics.csv`; `elbow_curve_{2d,full}.csv` written per run.
- [x] **Real-world dataset** -- applied to UCI 529 Early Stage Diabetes (Islam et al. 2020). `configs/diabetes_positives.yaml` clusters the positive cohort (Cooper applied formulation) and `configs/diabetes_all.yaml` the full sample; outputs in `results/diabetes_positives/` and `results/diabetes_all/`.
