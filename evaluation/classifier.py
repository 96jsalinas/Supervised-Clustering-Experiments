"""Reusable classifier fit + evaluation primitive.

Shared by `PipelineRunner` (which needs classifier-level metrics alongside
the cluster-level ones) and `pipeline.tuning` (which scores candidate
hyperparameter combinations during CV).
"""

from time import perf_counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def compute_classifier_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Classifier metrics on a held-out partition.

    Binary: AUC via positive-class probability.
    Multiclass: AUC via one-vs-rest macro.
    Missing inputs (`None` or empty) yield NaN metrics.
    """
    if proba is None or y_true is None or len(y_true) == 0:
        return {
            "classifier_accuracy": np.nan,
            "classifier_auc": np.nan,
            "classifier_f1_macro": np.nan,
            "classifier_log_loss": np.nan,
        }
    y_pred = proba.argmax(axis=1)
    n_classes = proba.shape[1]
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro"
            )
    except ValueError:
        auc = np.nan
    try:
        ll = log_loss(y_true, proba, labels=list(range(n_classes)))
    except ValueError:
        ll = np.nan
    return {
        "classifier_accuracy": accuracy_score(y_true, y_pred),
        "classifier_auc": auc,
        "classifier_f1_macro": f1_score(y_true, y_pred, average="macro"),
        "classifier_log_loss": ll,
    }


def cross_val_report(
    model_cls,
    model_cfg: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> dict:
    """Stratified k-fold CV accuracy and AUC over the full dataset.

    Reports the classifier's performance the way the published baselines do
    (Islam et al. 2020 and Cooper et al. 2021 both use 10-fold CV), so the
    number is directly comparable rather than a single-split point estimate.
    The model is re-instantiated per fold from `model_cfg` (which should already
    carry any tuned params). Binary AUC uses the positive-class probability.

    Returns a dict of cv_* scalars suitable for merging into a metrics row.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    accs, aucs = [], []
    for tr, va in skf.split(X, y):
        model = model_cls(model_cfg)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[va])
        accs.append(accuracy_score(y[va], proba.argmax(axis=1)))
        if proba.shape[1] == 2:
            try:
                aucs.append(roc_auc_score(y[va], proba[:, 1]))
            except ValueError:
                pass

    report = {
        "cv_n_splits": n_splits,
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
    }
    if aucs:
        report["cv_auc_mean"] = float(np.mean(aucs))
        report["cv_auc_std"] = float(np.std(aucs))
    return report


def fit_and_eval_classifier(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Fit `model` on train, predict proba on both partitions, score on test.

    The caller owns the model instance (so `PipelineRunner` can keep using
    the fitted model for attribution afterwards). Returns a dict with the
    train/test probabilities, classifier metrics, and fit wall-clock time.
    """
    t0 = perf_counter()
    model.fit(X_train, y_train)
    time_model_fit = perf_counter() - t0

    proba_train = model.predict_proba(X_train)
    proba_test = model.predict_proba(X_test)
    metrics = compute_classifier_metrics(y_test, proba_test)

    return {
        "proba_train": proba_train,
        "proba_test": proba_test,
        "time_model_fit": time_model_fit,
        **metrics,
    }
