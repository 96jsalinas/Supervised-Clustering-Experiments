# Supervised Clustering Experiments

Systematic evaluation of a three-step **supervised clustering** pipeline for subgroup discovery, as part of Josu Salinas's Master's Thesis (TFM) at UC3M.

The pipeline takes raw tabular data through four stages:

1. **Model training** -- train a supervised classifier. The baseline is an MLP (PyTorch); LightGBM is available as an alternative.
2. **Feature attribution** -- use the trained model to extract per-sample feature importances (e.g. SHAP, LRP). The model is shared across attribution methods so comparisons are fair.
3. **Dimensionality reduction** -- project the attribution matrix to 2D (e.g. UMAP).
4. **Clustering** -- identify subgroups in the reduced space (e.g. DBSCAN, HDBSCAN).

The framework is config-driven: each experiment is defined by a YAML file specifying the method and hyperparameters for every step. Adding a new method requires one Python file and one line in the registry.

Core methodology reference: Cooper, A. (2022). [Supervised Clustering with SHAP Values](https://www.aidancooper.co.uk/supervised-clustering-shap-values/).

## Repository structure

```
.
├── run_experiment.py           # Single-config CLI entry point
├── configs/                    # YAML experiment configs (single runs)
│   ├── cooper_dbscan.yaml      # Cooper blogpost reproduction (sanity check, LightGBM)
│   ├── hdbscan_baseline.yaml   # LightGBM+SHAP+UMAP+HDBSCAN reference
│   ├── mlp_baseline.yaml       # Thesis baseline: MLP+SHAP+UMAP+HDBSCAN
│   └── mlp_lrp.yaml            # MLP+LRP+UMAP+HDBSCAN
├── batch/                      # Cartesian-product sweeps over pipeline methods
│   ├── sweep.py                # Sweep runner (datasets x models x attr x red x clust)
│   ├── full_grid.yaml          # Method-only grid (one model, one dataset)
│   ├── robustness_grid.yaml    # Prior stability sweep (single model, seed + difficulty axes)
│   └── full_comparison_grid.yaml  # Overnight grid: two models, two generator families
├── pipeline/                   # Modular pipeline components
│   ├── base.py                 # Abstract base classes
│   ├── registry.py             # Maps config method names to Python classes
│   ├── runner.py               # Orchestrates model -> attribution -> reduction -> clustering
│   ├── models/                 # MLP, LightGBM
│   ├── attribution/            # SHAP, LRP, LIME
│   ├── reduction/              # UMAP, PCA, t-SNE, PaCMAP
│   └── clustering/             # DBSCAN, HDBSCAN, k-means
├── data/
│   └── synthetic.py            # Data generation via sklearn.make_blobs
├── evaluation/
│   ├── metrics.py              # External (ARI, NMI, AMI) and internal metrics + timings
│   ├── figures.py              # Per-run scatter / importance / per-cluster profile plots
│   └── dashboard.py            # Cross-run dashboard (metrics table + embedding grid + metric bars)
└── results/                    # Auto-created per run
    └── <config_name>/
        ├── config.yaml         # Verbatim copy of the config used
        ├── metrics.csv         # All computed metrics (including per-step wall-clock time)
        ├── arrays.npz          # Embedding + cluster labels, for cross-run dashboard
        └── figures/*.png       # All generated plots
```

## Quick start

See [SETUP.md](SETUP.md) for full setup instructions. In short:

```bash
pip install lightgbm shap umap-learn hdbscan scikit-learn pandas matplotlib seaborn pyyaml torch zennit lime pacmap
python run_experiment.py configs/mlp_baseline.yaml
```

Results appear in `results/mlp_baseline/`.

To sweep the full method grid:

```bash
python -m batch.sweep batch/full_grid.yaml
```

The sweep runner also accepts an optional `datasets:` list (iterate over data-config overrides) and an optional `models:` list (iterate over model configs) so a single spec can produce a (dataset × model × attribution × reduction × clustering) Cartesian product. Classifier tuning declared under `model.tune.enabled` is hoisted to run once per (dataset × model) cell and its winner propagates to every method combo for that cell. See `batch/full_comparison_grid.yaml` for the full-shape example and `batch/full_comparison_smoke.yaml` for a minimal version.

To aggregate all runs in `results/` into cross-run figures and a summary table:

```bash
python -m evaluation.dashboard --out figures/dashboard/
```

This produces `metrics_table.csv` / `metrics_table.png`, `embedding_grid.png` (one scatter per run), and `metric_bars.png` (external metrics grouped by pipeline-step variation) for direct LaTeX inclusion.

## Data generation

The synthetic datasets (`data/synthetic.py`) are produced by `sklearn.make_blobs` in an `n_informative`-dimensional subspace and padded with independent Gaussian noise features to reach `n_features` total. Binary class labels are derived as `y_class = y_subcluster % n_classes`, so cluster 0 → class 0, cluster 1 → class 1, cluster 2 → class 0, etc. `center_box` controls the range in which cluster centres are placed; narrowing it increases raw-feature-space overlap so the pipeline gets a non-trivial subgroup-discovery problem rather than clusters that separate on any single axis.

### Optional orthogonal rotation of the informative subspace

The `rotate_informative: true` flag in a `data:` block applies one extra step after `make_blobs` returns: the informative columns of `X` are multiplied by a random orthogonal matrix `Q` (QR-decomposed from a Gaussian matrix seeded by `random_state`). This is a deliberate, transparent way to test whether a model's performance depends on cluster centres being axis-aligned with the feature basis.

**What is preserved exactly** (verifiable with an equality check):

- Cluster identities (`y_subcluster` is untouched; rotation is a bijection).
- Per-cluster sample counts.
- All pairwise centroid-to-centroid distances (rotation is isometric).
- Isotropic within-cluster covariance.

**What changes:** cluster principal axes are no longer parallel to the feature axes, so splits along a single coordinate (the move tree ensembles rely on) no longer align with the informative structure. The MLP, which processes feature vectors through a learned linear layer, is basis-invariant and should be unaffected — any gap that persists or closes between models under rotation is evidence about axis-alignment, not about model capacity.

This is **not** an attempt to reimplement `sklearn.make_classification`. Centroid placement, feature correlations, redundant features, and label noise are all left exactly as the default `make_blobs` path produces them. Only the basis of the informative subspace is randomised.

## Current status

All pipeline methods (MLP, LightGBM, SHAP, LRP, LIME, UMAP, PCA, t-SNE, PaCMAP, DBSCAN, HDBSCAN, k-means) are implemented. Batch sweep accepts dataset and model axes; classifier-level metrics land in every run via a stratified train/test split; `model.tune` blocks can tune classifier hyperparameters per (dataset × model) cell via stratified K-fold CV. The cross-run dashboard produces model-aware pivot heatmaps, stability strips, and a colour-coded metrics table. See [TODO.md](TODO.md) for outstanding items.
