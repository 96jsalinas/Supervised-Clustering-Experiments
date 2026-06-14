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


# Human-readable labels for the encoded levels, keyed by the same mapping
# object used to encode. Carried through to cluster descriptions so a decision
# rule reads "polyuria=Yes" rather than "polyuria=1" and gender is not mislabelled
# as a Yes/No flag.
_LABELS = {
    id(_YESNO): {0: "No", 1: "Yes"},
    id(_GENDER): {0: "Female", 1: "Male"},
    id(_POSNEG): {0: "Negative", 1: "Positive"},
}


def _encode_categorical(series: pd.Series) -> tuple[np.ndarray, dict]:
    """Map a categorical column to 0/1, raising on any unmapped value.

    Picks the mapping by inspecting the column's value set rather than its name,
    so the loader does not hard-code which column is gender vs a Yes/No flag.

    Returns the encoded array and the inverse label map ({0: ..., 1: ...}) for
    the mapping that matched, so downstream code can render the original level
    names.
    """
    vals = series.astype(str).str.strip().str.lower()
    unique = set(vals.unique())
    for mapping in (_YESNO, _GENDER, _POSNEG):
        if unique <= set(mapping):
            encoded = vals.map(mapping).to_numpy(dtype=int)
            return encoded, _LABELS[id(mapping)]
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
      meta           : dict with 'positive_mask' (y_class == 1), 'n_positive',
                       and the inverse-transform metadata 'numeric_stats'
                       (per-numeric-column mean/std) and 'value_labels'
                       (per-categorical-column {0,1} -> level name), used by the
                       cluster-description step to report ages in years and
                       rules in the original Yes/No/Male-Female vocabulary.

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

    y_class, _ = _encode_categorical(df[target_col])

    feature_cols = [c for c in df.columns if c != target_col]
    columns = []
    # numeric_stats and value_labels let cluster_description.py undo the
    # standardisation (report age in years, not z-scores) and render decision
    # rules in the original level names. Recording the mean/std here, at the
    # single point where standardisation happens, keeps the inverse transform
    # honest;
    numeric_stats: dict = {}
    value_labels: dict = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            x = df[col].to_numpy(dtype=float)
            mean, std = float(x.mean()), float(x.std())
            numeric_stats[col] = {"mean": mean, "std": std}
            x = (x - mean) / std if std > 0 else x - mean
            columns.append(x)
        else:
            encoded, labels = _encode_categorical(df[col])
            value_labels[col] = labels
            columns.append(encoded.astype(float))

    X = np.column_stack(columns)

    positive_mask = y_class == 1
    meta = {
        "dataset": dataset,
        "positive_mask": positive_mask,
        "n_positive": int(positive_mask.sum()),
        "n_total": int(len(y_class)),
        # Inverse-transform metadata for interpretable cluster descriptions.
        "numeric_stats": numeric_stats,   # {col: {"mean", "std"}}
        "value_labels": value_labels,     # {col: {0: name, 1: name}}
    }

    return X, y_class, None, feature_cols, meta
