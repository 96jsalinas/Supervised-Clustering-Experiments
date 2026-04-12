# Setup and usage

## Requirements

- Python 3.10+
- The packages listed in the install command below (no `requirements.txt` yet, as the dependency set is small and stable)

## Installation

From the repository root:

```bash
pip install lightgbm shap umap-learn hdbscan scikit-learn pandas matplotlib seaborn pyyaml
```

Tested with the following versions (other recent versions should also work):

| Package       | Version |
|---------------|---------|
| lightgbm      | 4.6.0   |
| shap          | 0.51.0  |
| umap-learn    | 0.5.12  |
| hdbscan       | 0.8.42  |
| scikit-learn  | 1.6.1   |
| pandas        | 2.2.3   |
| matplotlib    | 3.10.0  |
| seaborn       | 0.13.2  |
| PyYAML        | 6.0.2   |

## Running an experiment

```bash
python run_experiment.py configs/cooper_dbscan.yaml
```

The script resolves all paths relative to its own location, so it can be called from any working directory. Output goes to `results/<config_name>/`.

## Available configs

| Config                      | Purpose                                       |
|-----------------------------|-----------------------------------------------|
| `cooper_dbscan.yaml`        | Reproduce Cooper's blogpost as a sanity check |
| `hdbscan_baseline.yaml`     | Official thesis baseline (HDBSCAN)            |
| `r_reference_example1.yaml` | Reproduce supervisor's R reference (Example I)|

## Creating a new experiment

1. Copy an existing YAML config and modify the method names and parameters.
2. Run it with `python run_experiment.py configs/your_config.yaml`.
3. Results appear in `results/your_config/`.

Method names available in the config (registered in `pipeline/registry.py`):

| Step          | Implemented           | Stubs (not yet working)       |
|---------------|-----------------------|-------------------------------|
| Model         | `lightgbm`            | `mlp`                         |
| Attribution   | `shap`                | `lrp`, `lime`                 |
| Reduction     | `umap`                | `pca`, `tsne`, `pacmap`       |
| Clustering    | `dbscan`, `hdbscan`   | `kmeans`                      |

Note that `model` and `attribution` are separate config sections. The model is trained once and passed to the attribution method, so SHAP and LRP can be evaluated on the same model for a fair comparison.

## Output structure

Each run produces:

- `config.yaml` -- exact copy of the config used, for reproducibility.
- `metrics.csv` -- external (ARI, NMI, AMI vs true subclusters) and internal (Silhouette, Davies-Bouldin, Calinski-Harabasz) metrics. Two rows: one for clustering in the 2D embedding, one for clustering in the full attribution space (no DR).
- `figures/` -- PNG plots ready for LaTeX inclusion:
  - `umap_raw_true_labels.png` -- UMAP of raw features colored by class.
  - `umap_shap_true_labels.png` -- UMAP of SHAP values colored by class.
  - `umap_shap_cluster_labels.png` -- UMAP of SHAP values colored by predicted cluster.
  - `shap_importance_bar.png` -- mean absolute SHAP value per feature.
