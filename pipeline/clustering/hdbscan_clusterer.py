from itertools import product

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

from pipeline.base import BaseClusterer


class HDBSCANClusterer(BaseClusterer):
    """Hierarchical density-based clustering via HDBSCAN.

    Two modes, mutually exclusive:
    - Fixed params: set ``params`` in the config (back-compatible default).
    - Auto-select via grid search: set ``auto_select.enabled: true``. Mirrors
      Cooper et al. (2021), who grid-search min_samples / min_cluster_size /
      cluster_selection_epsilon and pick the configuration that maximises the
      mean silhouette subject to a cap on the fraction of unassigned (noise)
      points.

    After fit_predict the attributes ``selected_k_`` (cluster count of the chosen
    configuration) and ``elbow_df_`` (the full search grid) are always set, so
    the runner can harvest them exactly as it does for the k-means clusterer.
    """

    selected_k_: "int | None" = None
    elbow_df_: "pd.DataFrame | None" = None
    selected_params_: "dict | None" = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        auto_cfg = self.config.get("auto_select") or {}
        if auto_cfg.get("enabled", False):
            return self._auto_select(X, auto_cfg)

        self.selected_k_ = None
        self.elbow_df_ = None
        self.selected_params_ = None
        params = self.config.get("params", {})
        return HDBSCAN(**params).fit_predict(X)

    @staticmethod
    def _n_clusters(labels: np.ndarray) -> int:
        return len(set(labels[labels >= 0]))

    def _auto_select(self, X: np.ndarray, auto_cfg: dict) -> np.ndarray:
        """Grid-search HDBSCAN params, maximising silhouette under a noise cap.

        The objective and constraint follow Cooper et al. (2021): pick the
        configuration with the highest mean silhouette such that no more than
        ``max_noise_fraction`` of points are left unassigned. If no configuration
        satisfies the noise cap, fall back to the highest-silhouette config among
        those that produced at least two clusters, recording that the cap was
        relaxed in ``elbow_df_``.
        """
        mcs_grid = auto_cfg.get("min_cluster_size", [5, 10, 15, 20, 25])
        ms_grid = auto_cfg.get("min_samples", [None])
        eps_grid = auto_cfg.get("cluster_selection_epsilon", [0.0])
        max_noise = float(auto_cfg.get("max_noise_fraction", 0.03))

        n = len(X)
        rows = []
        results = []  # (silhouette, n_clusters, noise_frac, params, labels)
        for mcs, ms, eps in product(mcs_grid, ms_grid, eps_grid):
            kwargs = {
                "min_cluster_size": int(mcs),
                "cluster_selection_epsilon": float(eps),
            }
            if ms is not None:
                kwargs["min_samples"] = int(ms)
            labels = HDBSCAN(**kwargs).fit_predict(X)

            mask = labels >= 0
            n_clusters = self._n_clusters(labels)
            noise_frac = float((~mask).sum() / n)
            if n_clusters >= 2:
                sil = float(silhouette_score(X[mask], labels[mask]))
            else:
                sil = float("nan")

            rows.append({
                "min_cluster_size": int(mcs),
                "min_samples": (int(ms) if ms is not None else None),
                "cluster_selection_epsilon": float(eps),
                "n_clusters": n_clusters,
                "noise_fraction": noise_frac,
                "silhouette": sil,
                "feasible": bool(n_clusters >= 2 and noise_frac <= max_noise),
            })
            results.append((sil, n_clusters, noise_frac, kwargs, labels))

        self.elbow_df_ = pd.DataFrame(rows)

        def _pick(predicate):
            best = None
            for sil, k, noise, kwargs, labels in results:
                if np.isnan(sil) or not predicate(k, noise):
                    continue
                if best is None or sil > best[0]:
                    best = (sil, k, kwargs, labels)
            return best

        # Primary: silhouette-maximal under the noise cap. Fallback: drop the cap.
        best = _pick(lambda k, noise: k >= 2 and noise <= max_noise)
        self.elbow_df_.attrs["noise_cap_relaxed"] = False
        if best is None:
            best = _pick(lambda k, noise: k >= 2)
            self.elbow_df_.attrs["noise_cap_relaxed"] = True

        if best is None:
            # No configuration produced >= 2 clusters: return the last labels
            # (all noise / single cluster) so the pipeline degrades gracefully.
            self.selected_k_ = self._n_clusters(results[-1][4])
            self.selected_params_ = results[-1][3]
            return results[-1][4]

        self.selected_k_ = best[1]
        self.selected_params_ = best[2]
        return best[3]
