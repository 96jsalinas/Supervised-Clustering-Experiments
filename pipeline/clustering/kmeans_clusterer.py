import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator

from pipeline.base import BaseClusterer


class KMeansClusterer(BaseClusterer):
    """Centroid-based clustering via k-means.

    Supports two modes, mutually exclusive:
    - Fixed k: set params.n_clusters in the config.
    - Auto-select via elbow (kneedle): set auto_select.enabled: true with
      optional k_min / k_max bounds.

    After fit_predict, the attributes selected_k_ and elbow_df_ are always
    set: selected_k_ is the k that was used; elbow_df_ is a DataFrame with
    columns [k, inertia] covering the full search range (None in fixed-k mode
    since no curve was computed).
    """

    selected_k_: int | None = None
    elbow_df_: "pd.DataFrame | None" = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        random_state = self.config.get("random_state", None)
        params = self.config.get("params", {})
        auto_cfg = self.config.get("auto_select") or {}

        has_fixed_k = "n_clusters" in params
        has_auto = bool(auto_cfg.get("enabled", False))

        if has_fixed_k and has_auto:
            raise ValueError(
                "k-means config: set either params.n_clusters or "
                "auto_select.enabled, not both."
            )

        if has_auto:
            k_min = int(auto_cfg.get("k_min", 2))
            k_max = int(auto_cfg.get("k_max", 15))
            n_init = int(params.get("n_init", 10))

            ks = list(range(k_min, k_max + 1))
            inertias = [
                KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
                .fit(X).inertia_
                for k in ks
            ]

            knee = KneeLocator(
                ks, inertias, curve="convex", direction="decreasing",
                S=auto_cfg.get("S", 1.0),
            )
            selected_k = knee.knee if knee.knee is not None else k_min

            self.selected_k_ = selected_k
            self.elbow_df_ = pd.DataFrame({"k": ks, "inertia": inertias})

            remaining = {k: v for k, v in params.items() if k not in ("n_clusters", "n_init")}
            return KMeans(
                n_clusters=selected_k, n_init=n_init,
                random_state=random_state, **remaining,
            ).fit_predict(X)

        # Fixed-k path
        self.selected_k_ = params.get("n_clusters")
        self.elbow_df_ = None
        return KMeans(random_state=random_state, **params).fit_predict(X)
