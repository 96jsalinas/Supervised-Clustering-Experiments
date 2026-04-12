# Supervised Clustering Experiments

Systematic evaluation of a three-step **supervised clustering** pipeline for subgroup discovery, as part of Josu Salinas's Master's Thesis (TFM) at UC3M.

The pipeline takes raw tabular data through four stages:

1. **Model training** -- train a supervised classifier (e.g. LightGBM, MLP).
2. **Feature attribution** -- use the trained model to extract per-sample feature importances (e.g. SHAP, LRP). The model is shared across attribution methods so comparisons are fair.
3. **Dimensionality reduction** -- project the attribution matrix to 2D (e.g. UMAP).
4. **Clustering** -- identify subgroups in the reduced space (e.g. DBSCAN, HDBSCAN).

The framework is config-driven: each experiment is defined by a YAML file specifying the method and hyperparameters for every step. Adding a new method requires one Python file and one line in the registry.

Core methodology reference: Cooper, A. (2022). [Supervised Clustering with SHAP Values](https://www.aidancooper.co.uk/supervised-clustering-shap-values/).

## Repository structure

```
.
├── run_experiment.py           # CLI entry point
├── configs/                    # YAML experiment configs
│   ├── cooper_dbscan.yaml      # Cooper blogpost reproduction (sanity check)
│   └── hdbscan_baseline.yaml   # Official thesis baseline
├── pipeline/                   # Modular pipeline components
│   ├── base.py                 # Abstract base classes
│   ├── registry.py             # Maps config method names to Python classes
│   ├── runner.py               # Orchestrates model -> attribution -> reduction -> clustering
│   ├── models/                 # LightGBM (implemented), MLP (stub)
│   ├── attribution/            # SHAP (implemented), LRP & LIME (stubs)
│   ├── reduction/              # UMAP (implemented), PCA, t-SNE & PaCMAP (stubs)
│   └── clustering/             # DBSCAN & HDBSCAN (implemented), k-means (stub)
├── data/
│   └── synthetic.py            # Data generation via sklearn.make_classification
├── evaluation/
│   ├── metrics.py              # External (ARI, NMI, AMI) and internal metrics
│   └── figures.py              # UMAP scatter plots, SHAP importance bar chart
└── results/                    # Auto-created per run
    └── <config_name>/
        ├── config.yaml         # Verbatim copy of the config used
        ├── metrics.csv         # All computed metrics
        └── figures/*.png       # All generated plots
```

## Quick start

See [SETUP.md](SETUP.md) for full setup instructions. In short:

```bash
pip install lightgbm shap umap-learn hdbscan scikit-learn pandas matplotlib seaborn pyyaml
python run_experiment.py configs/cooper_dbscan.yaml
```

Results appear in `results/cooper_dbscan/`.

## Current status

The baseline pipeline (LightGBM + SHAP + UMAP + DBSCAN/HDBSCAN) is fully functional. Multiple alternative methods are planned but not yet implemented. See [TODO.md](TODO.md) for the full list of outstanding work.
