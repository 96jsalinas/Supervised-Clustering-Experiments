"""Qualitative description of discovered clusters in original feature units.

1. ``cluster_prevalence_table`` -- per-cluster mean of every feature. For the
   binary symptom/gender flags this is a prevalence (fraction of the cluster
   with the symptom); for age it is the mean age in years.
2. ``fit_onevsall_trees`` + ``onevsall_rules_table`` + ``save_onevsall_tree_figure``
   -- one shallow "cluster vs rest" ``DecisionTreeClassifier`` per cluster, fitted
   on the raw features, following Cooper's one-vs-all SkopeRules step as closely
   as the pinned stack allows (a plain sklearn tree stands in for SkopeRules; see
   SETUP.md). ``max_depth=2`` matches Cooper's two-comparison-term constraint. One
   best rule per cluster is reported with precision and recall, and the per-cluster
   trees are drawn as a grid figure. Cooper builds these rules on the raw features
   on purpose, characterising the clusters independently of the SHAP values that
   produced them.

Original units, not z-scores
----------------------------
The pipeline standardises numeric columns (data/real.py). Descriptions here run
on the *de-standardised* feature matrix so a rule reads "age > 47.5" and the
prevalence table shows ages in years; ``numeric_stats`` and ``value_labels``
from the loader's ``meta`` drive that inverse transform. Binary flags are left
as 0/1 for the prevalence mean (so the value is a readable fraction) and
rendered with their original level names in the rules.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier


def interpretable_features(
    X_sub: np.ndarray, feature_names: list[str], numeric_stats: dict
) -> np.ndarray:
    """De-standardise numeric columns back to their original units.

    Binary columns are untouched (already 0/1). Numeric columns named in
    ``numeric_stats`` are mapped z -> z*std + mean. Returns a copy.
    """
    X = X_sub.astype(float).copy()
    for col, stats in (numeric_stats or {}).items():
        if col not in feature_names:
            continue
        j = feature_names.index(col)
        X[:, j] = X[:, j] * stats["std"] + stats["mean"]
    return X


def cluster_prevalence_table(
    X_disp: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray,
    numeric_cols: set,
) -> pd.DataFrame:
    """Per-cluster mean of every feature, plus an ``n`` count and overall row.

    For binary features the mean is the prevalence within the cluster; for
    numeric features (e.g. age) it is the mean in original units. Noise points
    (label -1) get their own row when present. The trailing ``overall`` row is
    the whole clustered subset, for contrast.
    """
    rows = {}
    order = [c for c in sorted(np.unique(labels)) if c != -1]
    if -1 in labels:
        order.append(-1)
    for c in order:
        mask = labels == c
        name = "noise" if c == -1 else f"cluster_{c}"
        means = X_disp[mask].mean(axis=0)
        rows[name] = dict(zip(feature_names, means), n=int(mask.sum()))
    rows["overall"] = dict(
        zip(feature_names, X_disp.mean(axis=0)), n=int(len(labels))
    )
    table = pd.DataFrame.from_dict(rows, orient="index")
    table = table[[*feature_names, "n"]]
    # Round for readability: binary prevalences to 2 dp, ages to 1 dp.
    for col in feature_names:
        table[col] = table[col].round(1 if col in numeric_cols else 2)
    return table


def _condition(feat: str, thresh: float, go_left: bool,
               numeric_cols: set, value_labels: dict) -> str:
    """Render one split as a readable condition.

    ``go_left`` is the ``<= thresh`` branch. Numeric features keep the numeric
    threshold; binary features render as ``feat=<level name>`` using the
    loader's value labels (0 on the left branch, 1 on the right).
    """
    if feat in numeric_cols:
        return f"{feat} <= {thresh:.1f}" if go_left else f"{feat} > {thresh:.1f}"
    level = 0 if go_left else 1
    label = (value_labels.get(feat, {}) or {}).get(level, level)
    return f"{feat}={label}"


def fit_onevsall_trees(
    X_disp: np.ndarray,
    labels: np.ndarray,
    max_depth: int = 2,
    random_state: int = 42,
) -> tuple[dict, dict]:
    """Fit one shallow "cluster vs rest" tree per cluster (Cooper's method).

    Mirrors Cooper's SkopeRules step: for each cluster a separate
    ``DecisionTreeClassifier`` is fitted on the de-standardised raw features to
    tell that cluster apart from all the others, independently of the SHAP values
    that produced the clustering. ``max_depth=2`` matches Cooper's constraint of
    rules with at most two comparison terms. Noise points (label -1) are dropped.
    Returns the per-cluster trees and the per-cluster sample counts.
    """
    keep = labels != -1
    Xk, yk = X_disp[keep], labels[keep]
    clusters = [int(c) for c in sorted(np.unique(yk))]
    class_counts = {c: int((yk == c).sum()) for c in clusters}
    trees = {}
    for c in clusters:
        y = (yk == c).astype(int)
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        tree.fit(Xk, y)
        trees[c] = tree
    return trees, class_counts


def _best_onevsall_rule(tree: DecisionTreeClassifier, feature_names: list[str],
                        numeric_cols: set, value_labels: dict,
                        n_in_cluster: int):
    """Best single rule from a fitted cluster-vs-rest tree.

    Walks the tree's leaves, scores each leaf that captures cluster members by
    precision (purity) and recall (share of the cluster captured), and returns
    the one maximising F1 -- the balance Cooper's SkopeRules trades off when it
    keeps its top rule. Returns None if no leaf captures any cluster member.
    """
    t = tree.tree_
    candidates = []

    def recurse(node, conds):
        left, right = t.children_left[node], t.children_right[node]
        if left == right:  # leaf
            # tree_.value holds class proportions (summing to 1); recover counts
            # via n_node_samples. Index 1 is the positive (in-cluster) class.
            total = int(t.n_node_samples[node])
            value = t.value[node][0]
            s = value.sum()
            precision = float(value[1] / s) if s > 0 else 0.0
            n_correct = precision * total
            if n_correct > 0:
                recall = n_correct / n_in_cluster if n_in_cluster else 0.0
                f1 = (2 * precision * recall / (precision + recall)
                      if precision + recall > 0 else 0.0)
                candidates.append({
                    "rule": " AND ".join(conds) if conds else "(all)",
                    "precision": precision,
                    "recall": recall,
                    "n_rule": total,
                    "n_correct": int(round(n_correct)),
                    "_f1": f1,
                })
            return
        feat = feature_names[t.feature[node]]
        thresh = float(t.threshold[node])
        recurse(left, conds + [
            _condition(feat, thresh, True, numeric_cols, value_labels)])
        recurse(right, conds + [
            _condition(feat, thresh, False, numeric_cols, value_labels)])

    recurse(0, [])
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r["_f1"])
    best.pop("_f1")
    return best


def onevsall_rules_table(
    trees: dict,
    class_counts: dict,
    feature_names: list[str],
    numeric_cols: set,
    value_labels: dict,
) -> pd.DataFrame:
    """One best rule per cluster, with precision and recall (Cooper's Table 1).

    For each cluster-vs-rest tree, keeps the single best rule (by F1) and reports
    it alongside Cooper's two scores: precision (of patients matching the rule,
    the share truly in the cluster) and recall (of the cluster, the share the
    rule captures). ``count_pct`` is the cluster's share of the clustered sample,
    matching the "Count (%)" column of the paper's Table 1.
    """
    total = sum(class_counts.values())
    records = []
    for c, tree in trees.items():
        best = _best_onevsall_rule(
            tree, feature_names, numeric_cols, value_labels, class_counts[c]
        )
        if best is None:
            best = {"rule": "(none)", "precision": 0.0, "recall": 0.0,
                    "n_rule": 0, "n_correct": 0}
        records.append({
            "cluster": c,
            "n_cluster": class_counts[c],
            "count_pct": round(100 * class_counts[c] / total, 1) if total else 0.0,
            "rule": best["rule"],
            "precision": round(best["precision"], 3),
            "recall": round(best["recall"], 3),
            "n_rule": best["n_rule"],
            "n_correct": best["n_correct"],
        })
    return pd.DataFrame.from_records(records)


def save_rules_table_figure(
    rule_df: pd.DataFrame,
    figures_dir: Path,
    space_tag: str,
):
    """Render the one-vs-all rules as a Cooper-style table image (his Table 1).

    Columns mirror the paper's Table 1: cluster, Count (%), the rule, and
    precision/recall as percentages. This is the more legible companion to the
    tree grid; both describe the same rules.
    """
    show = pd.DataFrame({
        "cluster": rule_df["cluster"],
        "count": rule_df["n_cluster"].astype(str)
                 + " (" + rule_df["count_pct"].astype(str) + "%)",
        "rule": rule_df["rule"],
        "precision": (rule_df["precision"] * 100).round().astype(int).astype(str) + "%",
        "recall": (rule_df["recall"] * 100).round().astype(int).astype(str) + "%",
    })
    headers = ["Cluster", "Count (%)", "Decision rule (raw features)",
               "Precision", "Recall"]
    nrows = len(show)
    fig, ax = plt.subplots(figsize=(12, 0.5 * nrows + 1.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=show.values,
        colLabels=headers,
        colWidths=[0.08, 0.12, 0.55, 0.13, 0.12],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#e8e8e8")
    ax.set_title(f"One-vs-all cluster decision rules ({space_tag})", pad=12)
    fig.tight_layout()
    fig.savefig(figures_dir / f"cluster_rules_table_{space_tag}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def save_onevsall_tree_figure(
    trees: dict,
    feature_names: list[str],
    figures_dir: Path,
    space_tag: str,
):
    """Grid of the per-cluster "cluster vs rest" trees, one subplot per cluster.

    Binary splits render in scikit-learn's native "feature <= 0.5" form; the
    rules CSV carries the same splits in the cleaner "feature=Yes/No" wording.
    """
    from sklearn.tree import plot_tree

    clusters = list(trees.keys())
    n = len(clusters)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, c in enumerate(clusters):
        ax = axes[idx // ncols][idx % ncols]
        plot_tree(
            trees[c],
            feature_names=feature_names,
            class_names=["other", f"cluster_{c}"],
            filled=True,
            rounded=True,
            impurity=False,
            fontsize=7,
            ax=ax,
        )
        ax.set_title(f"Cluster {c} vs rest")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"One-vs-all cluster decision rules ({space_tag})")
    fig.tight_layout()
    fig.savefig(figures_dir / f"cluster_decision_tree_{space_tag}.png", dpi=150)
    plt.close(fig)


def save_prevalence_heatmap(
    table: pd.DataFrame,
    feature_names: list[str],
    numeric_cols: set,
    figures_dir: Path,
    space_tag: str,
):
    """Heatmap of binary-symptom prevalence per cluster (clusters x symptoms).

    Numeric columns (age) and the ``n`` count are dropped so every cell is a
    comparable 0-1 prevalence; the ``overall`` row is kept as a reference column
    of base rates. Cells are annotated with the prevalence.
    """
    binary_cols = [f for f in feature_names if f not in numeric_cols]
    if not binary_cols:
        return
    mat = table.loc[:, binary_cols]
    fig, ax = plt.subplots(
        figsize=(0.5 * len(binary_cols) + 3, 0.5 * len(mat) + 2)
    )
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(binary_cols)))
    ax.set_xticklabels(binary_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=6)
    ax.set_title(f"Symptom prevalence per cluster ({space_tag})")
    fig.colorbar(im, ax=ax, label="prevalence", fraction=0.03)
    fig.tight_layout()
    fig.savefig(figures_dir / f"cluster_symptom_prevalence_{space_tag}.png",
                dpi=150)
    plt.close(fig)


def _labelings_agreement(labels_2d: np.ndarray,
                         labels_full: np.ndarray) -> dict:
    """Compare the 2D-embedding and full-space clusterings on the same rows."""
    k2 = len([c for c in np.unique(labels_2d) if c != -1])
    kf = len([c for c in np.unique(labels_full) if c != -1])
    return {
        "n_clusters_2d": k2,
        "n_clusters_full": kf,
        "ari_2d_vs_full": float(adjusted_rand_score(labels_2d, labels_full)),
    }


def describe_clusters(
    result,
    feature_names: list[str],
    meta: dict,
    output_dir: Path,
    figures_dir: Path,
    max_depth: int = 2,
) -> dict:
    """Produce prevalence tables and Cooper-style one-vs-all cluster rules.

    For each labeling (2D embedding, full attribution space) writes the symptom
    prevalence table, fits one shallow "cluster vs rest" tree per cluster on the
    raw features (Cooper's SkopeRules step), and saves the per-cluster trees as a
    grid figure plus a rules CSV (one best rule per cluster, with precision and
    recall), all from the same trees. Also writes the 2D prevalence heatmap. The
    2D labeling is the reported one; full-space artefacts are kept for the record
    and the two are compared so a large divergence can be flagged. Returns a dict
    with the tables, rules and the agreement summary.
    """
    numeric_stats = (meta or {}).get("numeric_stats", {})
    value_labels = (meta or {}).get("value_labels", {})
    numeric_cols = set(numeric_stats)

    X_sub = result.X_raw[result._mask()]
    X_disp = interpretable_features(X_sub, feature_names, numeric_stats)

    labelings = {
        "2d": result.cluster_labels_2d,
        "full": result.cluster_labels_full,
    }
    tables, rules = {}, {}
    for tag, labels in labelings.items():
        table = cluster_prevalence_table(
            X_disp, feature_names, labels, numeric_cols
        )
        table.to_csv(output_dir / f"cluster_symptom_prevalence_{tag}.csv")
        trees, class_counts = fit_onevsall_trees(X_disp, labels, max_depth)
        rule_df = onevsall_rules_table(
            trees, class_counts, feature_names, numeric_cols, value_labels
        )
        rule_df.to_csv(
            output_dir / f"cluster_tree_rules_{tag}.csv", index=False
        )
        save_rules_table_figure(rule_df, figures_dir, tag)
        save_onevsall_tree_figure(trees, feature_names, figures_dir, tag)
        tables[tag], rules[tag] = table, rule_df

    save_prevalence_heatmap(
        tables["2d"], feature_names, numeric_cols, figures_dir, "2d"
    )

    agreement = _labelings_agreement(labelings["2d"], labelings["full"])
    return {"tables": tables, "rules": rules, "agreement": agreement,
            "tree_max_depth": max_depth}
