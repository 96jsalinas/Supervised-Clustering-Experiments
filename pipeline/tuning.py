"""Generic classifier hyperparameter tuning via stratified K-fold CV.

Model-agnostic: operates on any `BaseModel` subclass by re-instantiating it
with candidate `params` overrides and calling `.fit` / `.predict_proba`
through the standard interface. Called from `PipelineRunner.run()` when the
model config contains an enabled `tune:` block, and from the sweep layer
when tuning is hoisted to the dataset level.
"""

from copy import deepcopy
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def _score_accuracy(y_true, proba):
    return accuracy_score(y_true, proba.argmax(axis=1))


def _score_auc(y_true, proba):
    n_classes = proba.shape[1]
    if n_classes == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")


def _score_f1_macro(y_true, proba):
    return f1_score(y_true, proba.argmax(axis=1), average="macro")


def _score_neg_log_loss(y_true, proba):
    n_classes = proba.shape[1]
    return -log_loss(y_true, proba, labels=list(range(n_classes)))


# All scorers return a value where larger is better, so the selection logic
# can be "argmax across the grid" regardless of which metric the user picked.
SCORERS = {
    "accuracy": _score_accuracy,
    "auc": _score_auc,
    "f1_macro": _score_f1_macro,
    "log_loss": _score_neg_log_loss,
}


def _enumerate_grid(grid: dict) -> list[dict]:
    """Turn a {param: [values]} dict into a list of {param: value} dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _combo_tag(combo: dict) -> str:
    """Short, filename-safe tag for a parameter combination."""
    parts = []
    for k, v in combo.items():
        if isinstance(v, list):
            v_str = "x".join(str(x) for x in v)
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return "|".join(parts)


def tune_classifier(
    model_cls,
    base_config: dict,
    tune_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """Stratified K-fold CV hyperparameter search.

    Parameters
    ----------
    model_cls : class
        Any `BaseModel` subclass. Instantiated fresh per grid point × fold.
    base_config : dict
        The `model:` block from the pipeline config. `params` supplies
        defaults; grid overrides replace the corresponding keys.
    tune_cfg : dict
        `{enabled, cv: {n_splits, shuffle, random_state}, scoring, grid}`.
    X_train, y_train : arrays
        The *train* partition of the outer train/test split. Tuning must
        never see the held-out test partition.

    Returns
    -------
    selected : dict
        `{params, cv_score_mean, cv_score_std, scoring, combo_tag}`.
    grid_rows : DataFrame
        One row per (grid point × fold) with the score and fold fit time.
    """
    scoring = tune_cfg.get("scoring", "accuracy")
    if scoring not in SCORERS:
        raise ValueError(
            f"Unknown scoring '{scoring}'. Supported: {sorted(SCORERS)}"
        )
    scorer = SCORERS[scoring]

    cv_cfg = tune_cfg.get("cv", {}) or {}
    n_splits = int(cv_cfg.get("n_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    cv_random_state = cv_cfg.get("random_state", 42) if shuffle else None
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=cv_random_state
    )

    grid_combos = _enumerate_grid(tune_cfg.get("grid", {}))
    print(f"    grid points: {len(grid_combos)} × {n_splits} folds "
          f"= {len(grid_combos) * n_splits} fits")

    rows = []
    base_params = deepcopy(base_config.get("params", {}) or {})
    for gi, combo in enumerate(grid_combos, start=1):
        combo_params = {**base_params, **combo}
        combo_cfg = deepcopy(base_config)
        combo_cfg["params"] = combo_params
        tag = _combo_tag(combo)
        fold_scores = []
        for fold_idx, (tr, va) in enumerate(skf.split(X_train, y_train)):
            t0 = perf_counter()
            model = model_cls(combo_cfg)
            model.fit(X_train[tr], y_train[tr])
            proba = model.predict_proba(X_train[va])
            score = scorer(y_train[va], proba)
            dt = perf_counter() - t0
            fold_scores.append(score)
            rows.append({
                "grid_index": gi,
                "combo_tag": tag,
                "fold": fold_idx,
                "score": score,
                "scoring": scoring,
                "time_fit": dt,
                **{f"param_{k}": (tuple(v) if isinstance(v, list) else v)
                   for k, v in combo_params.items()},
            })
        mean = float(np.mean(fold_scores))
        std = float(np.std(fold_scores))
        print(f"    [{gi}/{len(grid_combos)}] {tag}  "
              f"{scoring}={mean:.4f}±{std:.4f}")

    grid_rows = pd.DataFrame(rows)

    # Winner: highest mean score across folds. All scorers are "larger is
    # better" (log_loss is negated), so argmax works uniformly.
    summary = (
        grid_rows.groupby("combo_tag")
        .agg(cv_score_mean=("score", "mean"),
             cv_score_std=("score", "std"))
        .reset_index()
    )
    best_row = summary.loc[summary["cv_score_mean"].idxmax()]
    best_tag = best_row["combo_tag"]
    winner_mask = (grid_rows["combo_tag"] == best_tag)
    def _native(v):
        """Coerce numpy / tuple types to yaml-safe Python natives."""
        if isinstance(v, tuple):
            return [_native(x) for x in v]
        if isinstance(v, list):
            return [_native(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v

    winner_params_full = {
        col.removeprefix("param_"): _native(
            grid_rows.loc[winner_mask, col].iloc[0]
        )
        for col in grid_rows.columns
        if col.startswith("param_")
    }

    selected = {
        "params": winner_params_full,
        "cv_score_mean": float(best_row["cv_score_mean"]),
        "cv_score_std": float(best_row["cv_score_std"]),
        "scoring": scoring,
        "combo_tag": best_tag,
    }
    return selected, grid_rows
