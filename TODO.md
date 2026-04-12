# TODO

Outstanding work for the experimental framework. Items are grouped by category.

## Pipeline methods to implement

- [ ] **MLP model** (`pipeline/models/mlp.py`) -- Multi-layer perceptron classifier. Required before LRP can be implemented, as LRP is defined for neural networks. Decide on PyTorch vs TensorFlow.
- [ ] **LRP attributor** (`pipeline/attribution/lrp.py`) -- Layer-wise Relevance Propagation. Receives a pre-trained MLP model via the shared model interface; no model training inside the attributor.
- [ ] **LIME attributor** (`pipeline/attribution/lime.py`) -- Local Interpretable Model-agnostic Explanations.
- [ ] **PCA reducer** (`pipeline/reduction/pca_reducer.py`) -- Linear baseline for dimensionality reduction.
- [ ] **t-SNE reducer** (`pipeline/reduction/tsne_reducer.py`) -- Nonlinear DR, useful as a comparison but poor for downstream clustering.
- [ ] **PaCMAP reducer** (`pipeline/reduction/pacmap_reducer.py`) -- Modern alternative to UMAP/t-SNE.
- [ ] **k-means clusterer** (`pipeline/clustering/kmeans_clusterer.py`) -- Weak baseline (assumes convex clusters).

## Evaluation and analysis

- [ ] **DBSCAN epsilon tuning** -- the Cooper config uses `eps=1.5`, which merges subclusters in our seeded UMAP embedding (finds 2 clusters instead of ~6). Either tune eps per embedding or add an automated eps estimation step (e.g. k-distance elbow).
- [ ] **UMAP of SHAP values colored by subcluster** -- currently the true-labels plot uses binary class labels. Add a figure colored by the 6 true subclusters for clearer visual validation.
- [ ] **Per-cluster SHAP profile plots** -- show which features drive each discovered cluster (waterfall or beeswarm per cluster).
- [ ] **Timing instrumentation** -- record wall-clock time per pipeline step for computational comparisons across methods.

## Infrastructure

- [ ] **Batch runner** -- script that runs all configs in `configs/` and produces a comparison table across runs.
- [ ] **`requirements.txt`** -- generate from the current working environment once the dependency set stabilizes.
- [ ] **Logging** -- replace `print()` statements with Python `logging` for configurable verbosity.

## Experiment design

- [ ] **Multiple synthetic datasets** -- vary `n_informative` and `n_clusters_per_class` as specified in the thesis methodology. Create configs for each scenario.
- [ ] **Stability analysis** -- run the same config with multiple seeds and report metric variance.
- [ ] **Real-world dataset** -- if time permits, apply the pipeline to a real dataset.
