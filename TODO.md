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
- **Dropped:** DBSCAN epsilon tuning -- DBSCAN was removed from the default grid (fixed `eps` is not a fair comparison vs HDBSCAN's auto-selection).

## Infrastructure

- [x] **Batch runner** (`batch/sweep.py`) -- Cartesian-product sweep with `--dry-run`, per-run result folders, shallow method-override merge, optional `datasets:` axis, optional `models:` axis, per-(dataset × model) classifier tuning hoisted out of the combo loop.
- [ ] **`requirements.txt`** -- generate from the current working environment once the dependency set stabilizes.
- [ ] **Logging** -- replace `print()` statements with Python `logging` for configurable verbosity.

## Experiment design

- [x] **Multiple synthetic datasets** -- `datasets:` axis in grid specs; `batch/robustness_grid.yaml` covers seed replicates + easy/medium/hard scenarios.
- [x] **Stability analysis** -- 24-run robustness sweep across 3 seeds and 3 difficulty scenarios; stability figure + pivot heatmaps in the dashboard.
- [x] **Second generator family (axis-alignment test)** -- `rotate_informative: true` applies a random orthogonal rotation to `make_blobs`'s informative subspace; preserves cluster identities and pairwise centroid distances exactly, breaks tree-friendly axis-alignment.
- [ ] **Full comparison sweep** -- `batch/full_comparison_grid.yaml` (42 dataset cells × 2 models × 4 method combos = 336 runs) queued for overnight execution. See `HANDOFF_OVERNIGHT_SWEEP.md` for run and interpretation instructions.
- [ ] **Real-world dataset** -- if time permits, apply the pipeline to a real dataset.
