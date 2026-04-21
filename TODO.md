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
- [x] **Cross-run dashboard** (`evaluation/dashboard.py`) -- aggregated metrics table, embedding grid, metric bars.
- [ ] **DBSCAN epsilon tuning** -- the Cooper config uses `eps=1.5`, which merges subclusters in our seeded UMAP embedding. Add automated eps estimation (e.g. k-distance elbow helper).
- [ ] **MLP hyperparameter tuning** -- close the ARI gap vs LightGBM (MLP baseline currently ~0.23 vs LightGBM ~0.41 on the same data). Required before M6 sweeps are meaningful.

## Infrastructure

- [x] **Batch runner** (`batch/sweep.py`) -- Cartesian-product sweep with `--dry-run`, per-run result folders, shallow method-override merge.
- [ ] **`requirements.txt`** -- generate from the current working environment once the dependency set stabilizes.
- [ ] **Logging** -- replace `print()` statements with Python `logging` for configurable verbosity.

## Experiment design

- [ ] **Multiple synthetic datasets** -- vary `n_informative` and `n_clusters` (and optionally `cluster_std`) to test different separation and complexity scenarios. Create configs for each scenario.
- [ ] **Stability analysis** -- run the same config with multiple seeds and report metric variance.
- [ ] **Real-world dataset** -- if time permits, apply the pipeline to a real dataset.
