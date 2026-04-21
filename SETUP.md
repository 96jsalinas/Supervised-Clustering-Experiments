# Setup and usage

## Requirements

- Python 3.10+
- The packages listed in the install command below (no `requirements.txt` yet, as the dependency set is small and stable)

## Installation

From the repository root:

```bash
pip install lightgbm shap umap-learn hdbscan scikit-learn pandas matplotlib seaborn pyyaml torch zennit lime pacmap
```

`torch` and `zennit` are required for the MLP baseline and LRP attributor; `lime` and `pacmap` for the corresponding methods.

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
| torch         | 2.x     |
| zennit        | 0.5.x   |
| lime          | 0.2.x   |
| pacmap        | 0.7.x   |

## Running an experiment

```bash
python run_experiment.py configs/cooper_dbscan.yaml
```

The script resolves all paths relative to its own location, so it can be called from any working directory. Output goes to `results/<config_name>/`.

## Available configs

| Config                      | Purpose                                       |
|-----------------------------|-----------------------------------------------|
| `cooper_dbscan.yaml`        | Reproduce Cooper's blogpost as a sanity check (LightGBM) |
| `hdbscan_baseline.yaml`     | LightGBM + SHAP + UMAP + HDBSCAN reference    |
| `mlp_baseline.yaml`         | Thesis baseline: MLP + SHAP + UMAP + HDBSCAN  |
| `mlp_lrp.yaml`              | MLP + LRP + UMAP + HDBSCAN                    |
| `r_reference_example1.yaml` | High cluster-separation scenario (cf. R reference Example I) |

## Batch sweeps

`batch/sweep.py` runs the Cartesian product of `attribution x reduction x clustering` methods defined in a grid spec. Each combination writes to `results/<attr>_<red>_<clust>/` with the same outputs as single runs.

```bash
python -m batch.sweep batch/full_grid.yaml --dry-run   # list combinations
python -m batch.sweep batch/full_grid.yaml             # run them
```

## Cross-run dashboard

After one or more runs exist under `results/`, aggregate them with:

```bash
python -m evaluation.dashboard --out figures/dashboard/
```

This walks every `results/<run>/` folder and produces `metrics_table.{csv,png}`, `embedding_grid.png` (one scatter per run, colored by discovered cluster), and `metric_bars.png` (external metrics grouped by pipeline-step variation) for direct LaTeX inclusion.

## Creating a new experiment

1. Copy an existing YAML config and modify the method names and parameters.
2. Run it with `python run_experiment.py configs/your_config.yaml`.
3. Results appear in `results/your_config/`.

Method names available in the config (registered in `pipeline/registry.py`):

| Step          | Implemented                              |
|---------------|------------------------------------------|
| Model         | `mlp` (baseline), `lightgbm`             |
| Attribution   | `shap`, `lrp`, `lime`                    |
| Reduction     | `umap`, `pca`, `tsne`, `pacmap`          |
| Clustering    | `dbscan`, `hdbscan`, `kmeans`            |

Note that `model` and `attribution` are separate config sections. The model is trained once and passed to the attribution method, so SHAP and LRP can be evaluated on the same model for a fair comparison.

## Output structure

Each run produces:

- `config.yaml` -- exact copy of the config used, for reproducibility.
- `metrics.csv` -- external (ARI, NMI, AMI vs true subclusters) and internal (Silhouette, Davies-Bouldin, Calinski-Harabasz) metrics, plus per-step wall-clock time. Two rows: clustering in the 2D embedding and in the full attribution space (no DR).
- `arrays.npz` -- the 2D embedding and cluster labels, consumed by `evaluation/dashboard.py`.
- `figures/` -- PNG plots ready for LaTeX inclusion:
  - `umap_raw_true_labels.png` -- 2D projection of raw features colored by class.
  - `umap_shap_true_labels.png` -- 2D projection of attributions colored by class.
  - `umap_shap_subcluster_labels.png` -- same projection colored by the 6 true subclusters.
  - `umap_shap_cluster_labels.png` -- same projection colored by predicted cluster.
  - `clusters_no_dr_vs_dr.png` -- side-by-side: clusters found in the full attribution space vs in the 2D embedding.
  - `shap_importance_bar.png` -- mean absolute attribution per feature.
  - `per_cluster_shap_profile.png` -- mean \|attribution\| per feature, one subplot per discovered cluster, sorted by global importance.
