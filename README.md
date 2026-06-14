# Supervised Clustering Experiments

Systematic evaluation of a four-step **supervised clustering** pipeline for subgroup discovery, as part of Josu Salinas's Master's Thesis (TFM) at UC3M.

The pipeline takes raw tabular data through four stages:

1. **Model training** -- train a supervised classifier. The baseline is an MLP (PyTorch); LightGBM is available as an alternative.
2. **Feature attribution** -- use the trained model to extract per-sample feature importances (e.g. SHAP, LRP). The model is shared across attribution methods so comparisons are fair.
3. **Dimensionality reduction** -- project the attribution matrix to 2D (e.g. UMAP).
4. **Clustering** -- identify subgroups in the reduced space (e.g. HDBSCAN, k-means with kneedle elbow selection, DBSCAN).

The framework is config-driven: each experiment is defined by a YAML file specifying the method and hyperparameters for every step. Adding a new method requires one Python file and one line in the registry.

Core methodology reference: Cooper, A. (2022). [Supervised Clustering with SHAP Values](https://www.aidancooper.co.uk/supervised-clustering-shap-values/).

## License and status

This code accompanies the Master's Thesis (TFM) of Josu Salinas at Universidad Carlos III de Madrid. The thesis has not yet been defended or published, so the repository is provisional and may change before the final version; it is shared in advance so the pipeline can be reused and extended. A citation will be added here once the thesis is published.

The code in this repository is released under the [MIT License](LICENSE). It depends on third-party libraries that keep their own licenses; all are permissive (MIT, BSD, Apache-2.0, PSF) with one exception: the LRP backend [Zennit](https://github.com/chr5tphr/zennit) is LGPL-3.0. Zennit is used as an unmodified, separately installed dependency, which the LGPL permits from MIT-licensed code. If you redistribute a bundle that includes Zennit itself, comply with the LGPL for that copy.

## Repository structure

```
.
├── run_experiment.py           # Single-config CLI entry point
├── configs/                    # YAML experiment configs (single runs)
│   ├── cooper_dbscan.yaml      # Cooper blogpost reproduction (sanity check, LightGBM)
│   ├── hdbscan_baseline.yaml   # LightGBM+SHAP+UMAP+HDBSCAN reference
│   ├── mlp_baseline.yaml       # Thesis baseline: MLP+SHAP+UMAP+HDBSCAN
│   ├── mlp_lrp.yaml            # MLP+LRP+UMAP+HDBSCAN
│   ├── diabetes_positives.yaml # Real data: cluster the positive cohort (Cooper applied)
│   └── diabetes_all.yaml       # Real data: cluster the full sample
├── batch/                      # Cartesian-product sweeps over pipeline methods
│   ├── sweep.py                # Sweep runner (datasets x models x attr x red x clust)
│   ├── sweep_a_data_variety.yaml   # Sweep A: data-variety, SHAP, MLP + LightGBM
│   ├── sweep_b_lrp_vs_shap.yaml    # Sweep B: LRP vs SHAP on the MLP
│   ├── sweep_b_lgbm_control.yaml   # Sweep B: LightGBM-SHAP control
│   ├── full_grid.yaml          # Method-only grid (one model, one dataset)
│   ├── robustness_grid.yaml    # Earlier stability sweep
│   └── full_comparison_grid.yaml   # Earlier overnight grid (pre-relaunch)
├── pipeline/                   # Modular pipeline components
│   ├── base.py                 # Abstract base classes
│   ├── registry.py             # Maps config method names to Python classes
│   ├── runner.py               # Orchestrates model -> attribution -> reduction -> clustering
│   ├── models/                 # MLP, LightGBM
│   ├── attribution/            # SHAP, LRP, LIME
│   ├── reduction/              # UMAP, PCA, t-SNE, PaCMAP
│   └── clustering/             # DBSCAN, HDBSCAN, k-means
├── data/
│   ├── synthetic.py            # Data generation via sklearn.make_blobs
│   └── real.py                 # Loader for documented real datasets (UCI 529)
├── evaluation/
│   ├── metrics.py              # External (ARI, NMI, AMI, F-measure) and internal metrics + timings
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

The synthetic datasets (`data/synthetic.py`) are produced by `sklearn.make_blobs` in an informative subspace and padded with independent Gaussian noise features to reach `n_features` total. The size of the informative subspace is set either by `n_informative` (an absolute count) or by `informative_fraction` (a fraction of `n_features`); the reported sweeps use `informative_fraction` so the signal-to-noise ratio stays fixed as `n_features` varies. Binary class labels are derived as `y_class = y_subcluster % n_classes`, so cluster 0 → class 0, cluster 1 → class 1, cluster 2 → class 0, etc. `center_box` controls the range in which cluster centres are placed; narrowing it increases raw-feature-space overlap so the pipeline gets a non-trivial subgroup-discovery problem rather than clusters that separate on any single axis.

### Optional orthogonal rotation of the informative subspace

The `rotate_informative: true` flag in a `data:` block applies one extra step after `make_blobs` returns: the informative columns of `X` are multiplied by a random orthogonal matrix `Q` (QR-decomposed from a Gaussian matrix seeded by `random_state`). This is a deliberate, transparent way to test whether a model's performance depends on cluster centres being axis-aligned with the feature basis.

**What is preserved exactly** (verifiable with an equality check):

- Cluster identities (`y_subcluster` is untouched; rotation is a bijection).
- Per-cluster sample counts.
- All pairwise centroid-to-centroid distances (rotation is isometric).
- Isotropic within-cluster covariance.

**What changes:** cluster principal axes are no longer parallel to the feature axes, so splits along a single coordinate (the move tree ensembles rely on) no longer align with the informative structure. The MLP, which processes feature vectors through a learned linear layer, is basis-invariant and should be unaffected — any gap that persists or closes between models under rotation is evidence about axis-alignment, not about model capacity.

This is **not** an attempt to reimplement `sklearn.make_classification`. Centroid placement, feature correlations, redundant features, and label noise are all left exactly as the default `make_blobs` path produces them. Only the basis of the informative subspace is randomised.

## Real datasets

A `data:` block with `source: real` loads a documented external dataset via `data/real.py` instead of generating synthetic data. Currently `dataset: uci529_early_diabetes` is supported (UCI ML Repository id 529, *Early Stage Diabetes Risk Prediction*, Islam et al. 2020), fetched with `ucimlrepo` and cached to `datasets/`. The loader z-score standardises numeric columns, maps the binary categoricals to 0/1, and returns no ground-truth subcluster labels.

Because real data has no ground-truth subgroups, the external recovery metrics (ARI, NMI, AMI, F-measure) are reported as `NaN` and the internal metrics (silhouette and the others) are primary. Example configs: `configs/diabetes_positives.yaml` and `configs/diabetes_all.yaml`.

### `clustering_subset`: where to cluster

A top-level `clustering_subset` key controls which rows reach dimensionality reduction and clustering, while the model and attribution stages always use the full sample:

- `all` (default) — reduce and cluster every sample, the convention used by the synthetic sweeps and Cooper's blogpost worked example.
- `positives` — reduce and cluster only the positive class (`y_class == 1`), replicating the applied formulation of Cooper et al. (2021), which trains and attributes on all participants but characterises subgroups within the positive cohort only.

## Current status

All pipeline methods (MLP, LightGBM, SHAP, LRP, LIME, UMAP, PCA, t-SNE, PaCMAP, DBSCAN, HDBSCAN, k-means) are implemented. Batch sweep accepts dataset and model axes; classifier-level metrics land in every run via a stratified train/test split; `model.tune` blocks can tune classifier hyperparameters per (dataset × model) cell via stratified K-fold CV. k-means selects its cluster count with the kneedle elbow, and the external metrics include the pair-counting F-measure alongside ARI, NMI, and AMI. Real tabular datasets are supported through a `source: real` data block together with the `clustering_subset` control. The cross-run dashboard produces model-aware pivot heatmaps, stability strips, and a colour-coded metrics table. See [TODO.md](TODO.md) for outstanding items.
