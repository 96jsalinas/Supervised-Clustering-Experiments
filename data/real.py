"""Loader for real, externally-documented datasets.

Parallel to data/synthetic.py. Where the synthetic generator fabricates
ground-truth subcluster identities, real data has none, so this loader returns
``y_subcluster = None``. Downstream code (metrics, figures) treats a None
subcluster array as "no ground truth": external recovery metrics (ARI/NMI/AMI/
F-measure) are skipped and internal metrics become primary.

Currently supports one dataset:

- ``uci529_early_diabetes`` -- UCI ML Repository id 529, "Early Stage Diabetes
  Risk Prediction" (Islam et al., 2020). 520 patients, 16 features (age, gender,
  14 Yes/No clinical signs), binary target ``class`` (Positive/Negative). No
  missing values.

The fetched frame is cached to ``datasets/uci529_early_diabetes.csv`` on first
use so the experiment is reproducible offline and the shared repo does not depend
on network access at run time.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

# Registry of supported real datasets: name -> (ucimlrepo id, target column,
# cache filename).
_REAL_DATASETS = {
    "uci529_early_diabetes": {
        "uci_id": 529,
        "target_col": "class",
        "cache": "uci529_early_diabetes.csv",
    },
}

# Case-insensitive value maps for the categorical columns. Binary symptom flags
# are Yes/No; gender is Male/Female; the diabetes target is Positive/Negative.
# Mapping is applied after lower-casing and stripping, and any value that does
# not map raises rather than silently becoming NaN.
_YESNO = {"yes": 1, "no": 0}
_GENDER = {"male": 1, "female": 0}
_POSNEG = {"positive": 1, "negative": 0}


def _encode_categorical(series: pd.Series) -> np.ndarray:
    """Map a categorical column to 0/1, raising on any unmapped value.

    Picks the mapping by inspecting the column's value set rather than its name,
    so the loader does not hard-code which column is gender vs a Yes/No flag.
    """
    vals = series.astype(str).str.strip().str.lower()
    unique = set(vals.unique())
    for mapping in (_YESNO, _GENDER, _POSNEG):
        if unique <= set(mapping):
            return vals.map(mapping).to_numpy(dtype=int)
    raise ValueError(
        f"Column '{series.name}': values {sorted(unique)} match none of the "
        f"known binary encodings (Yes/No, Male/Female, Positive/Negative)."
    )


def _fetch(spec: dict) -> pd.DataFrame:
    """Return the dataset frame, reading the local cache if present.

    On a cache miss, fetches via ucimlrepo, writes the cache, and returns it.
    """
    cache_path = DATASETS_DIR / spec["cache"]
    if cache_path.exists():
        return pd.read_csv(cache_path)

    from ucimlrepo import fetch_ucirepo  # local import: only needed on cache miss

    ds = fetch_ucirepo(id=spec["uci_id"])
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def load_real(
    data_config: dict,
) -> tuple[np.ndarray, np.ndarray, None, list[str], dict]:
    """Load and encode a real dataset for the supervised-clustering pipeline.

    Returns (X, y_class, y_subcluster, feature_names, meta) where:
      X              : (n_samples, n_features) float array, ready for the model.
      y_class        : (n_samples,) int array, target encoded to 0/1.
      y_subcluster   : always None -- real data has no ground-truth subgroups.
      feature_names  : column names in the order they appear in X.
      meta           : dict with 'positive_mask' (y_class == 1) and 'n_positive'.

    Encoding rules
    --------------
    - Numeric columns (e.g. age) are z-score standardised. This is harmless for
      the LightGBM baseline (scale-invariant) and keeps the MLP path valid.
      Standardisation uses full-dataset statistics; the tiny train/test leak this
      implies is immaterial for a scale-invariant tree baseline and is noted here
      rather than hidden.
    - Categorical columns are mapped to 0/1 via _encode_categorical, which raises
      on any value it cannot map (no silent NaN coercion).

    Config keys
    -----------
    dataset : registry name of the dataset. Default 'uci529_early_diabetes'.
    """
    dataset = data_config.get("dataset", "uci529_early_diabetes")
    if dataset not in _REAL_DATASETS:
        raise ValueError(
            f"Unknown real dataset '{dataset}'. "
            f"Known: {sorted(_REAL_DATASETS)}."
        )
    spec = _REAL_DATASETS[dataset]
    df = _fetch(spec)

    target_col = spec["target_col"]
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in {dataset}. "
            f"Columns: {list(df.columns)}."
        )

    # Guard against silent data corruption: this loader assumes a complete frame.
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"{dataset} has missing values in columns {bad}; this loader does "
            f"not impute. Inspect the source before proceeding."
        )

    y_class = _encode_categorical(df[target_col])

    feature_cols = [c for c in df.columns if c != target_col]
    columns = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            x = df[col].to_numpy(dtype=float)
            std = x.std()
            x = (x - x.mean()) / std if std > 0 else x - x.mean()
            columns.append(x)
        else:
            columns.append(_encode_categorical(df[col]).astype(float))

    X = np.column_stack(columns)

    positive_mask = y_class == 1
    meta = {
        "dataset": dataset,
        "positive_mask": positive_mask,
        "n_positive": int(positive_mask.sum()),
        "n_total": int(len(y_class)),
    }

    return X, y_class, None, feature_cols, meta
