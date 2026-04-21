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
│   ├── sweep.py                # Sweep runner
│   └── full_grid.yaml          # Grid spec (attribution x reduction x clustering)
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

To aggregate all runs in `results/` into cross-run figures and a summary table:

```bash
python -m evaluation.dashboard --out figures/dashboard/
```

This produces `metrics_table.csv` / `metrics_table.png`, `embedding_grid.png` (one scatter per run), and `metric_bars.png` (external metrics grouped by pipeline-step variation) for direct LaTeX inclusion.

## Current status

All pipeline methods (MLP, LightGBM, SHAP, LRP, LIME, UMAP, PCA, t-SNE, PaCMAP, DBSCAN, HDBSCAN, k-means) are implemented. Batch sweep and cross-run dashboard are in place. Next phase is experiment-design expansion (dataset variation, stability, DBSCAN eps tuning) and MLP hyperparameter tuning; see [TODO.md](TODO.md).
