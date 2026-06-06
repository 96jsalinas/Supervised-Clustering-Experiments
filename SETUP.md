# Setup and usage

## Requirements

- Python 3.10+
- The pinned dependency set in [`requirements.txt`](requirements.txt)

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` holds exact pins for every direct dependency plus the numerical
stack (numpy / numba / scipy / joblib). `torch` and `zennit` are required for the
MLP baseline and LRP attributor; `lime` and `pacmap` for the corresponding methods.

**Critical constraint:** numba requires `numpy < 2.2`, and UMAP and SHAP both run
on numba. Do not let an install pull numpy forward (for example, `pip install
imodels` upgrades numpy to 2.4 and breaks umap/shap). When adding a package, check
it does not move numpy or scikit-learn, or install it in a separate environment.
The pins exist because small drift in the numerical stack can perceptibly change
UMAP embeddings and therefore downstream clustering.

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
