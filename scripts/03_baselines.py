"""Phase 2 -- audit the cheap baselines before trusting any comparison.

Two questions this answers:

  1. Do the cheap descriptors carry real signal, or are they noise dressed up?
     Cross-validated scores under both split types, next to the random control.
  2. *Which* descriptors carry it?  Standardised ridge / logistic coefficients
     name the physical chemistry the model is leaning on, which is the part a
     wet-lab collaborator can argue with.

Writes results/baseline_scores.csv and results/baseline_coefficients.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lown.config import CHEN, RESULTS, TAP, TAP_TARGETS
from lown.data import load_chen, load_tap
from lown.features import build_matrix, feature_names, random_features
from lown.splits import identity_clusters

warnings.filterwarnings("ignore")

BLOCKS = ["random", "aac", "biophys", "cdr", "cheap"]


def _matrix(df, block):
    return random_features(df, 64, 0) if block == "random" else build_matrix(df, block)


def cv_scores(df, y, task, groups) -> pd.DataFrame:
    rows = []
    for block in BLOCKS:
        X = _matrix(df, block)
        if task == "classification":
            est = LogisticRegression(max_iter=3000, C=1.0)
            metric = "roc_auc"
            folds = {
                "random": StratifiedKFold(5, shuffle=True, random_state=0),
                "cluster": GroupKFold(5),
            }
        else:
            est = Ridge(alpha=10.0)
            metric = "r2"
            folds = {"random": KFold(5, shuffle=True, random_state=0), "cluster": GroupKFold(5)}
        pipe = make_pipeline(StandardScaler(), est)
        for split, cv in folds.items():
            s = cross_val_score(
                pipe, X, y, cv=cv, scoring=metric,
                groups=groups if split == "cluster" else None,
            )
            rows.append(
                dict(features=block, n_features=X.shape[1], split=split,
                     metric=metric, mean=float(s.mean()), std=float(s.std()))
            )
    return pd.DataFrame(rows)


def coefficients(df, y, task, top=12) -> pd.DataFrame:
    X = build_matrix(df, "cheap")
    names = feature_names("cheap")
    est = LogisticRegression(max_iter=3000) if task == "classification" else Ridge(alpha=10.0)
    pipe = make_pipeline(StandardScaler(), est).fit(X, y)
    coef = np.ravel(pipe[-1].coef_)
    order = np.argsort(-np.abs(coef))[:top]
    return pd.DataFrame(
        dict(feature=[names[i] for i in order], coefficient=coef[order].round(4))
    )


def main() -> None:
    chen, tap = load_chen(), load_tap()
    scores, coefs = [], []

    g = identity_clusters(chen)
    s = cv_scores(chen, chen.y.to_numpy(), "classification", g)
    s.insert(0, "target", "developability")
    s.insert(0, "dataset", CHEN)
    scores.append(s)
    c = coefficients(chen, chen.y.to_numpy(), "classification")
    c.insert(0, "target", "developability")
    c.insert(0, "dataset", CHEN)
    coefs.append(c)

    gt = identity_clusters(tap)
    for t in TAP_TARGETS:
        y = tap[t].to_numpy()
        s = cv_scores(tap, y, "regression", gt)
        s.insert(0, "target", t)
        s.insert(0, "dataset", TAP)
        scores.append(s)
        c = coefficients(tap, y, "regression")
        c.insert(0, "target", t)
        c.insert(0, "dataset", TAP)
        coefs.append(c)

    scores = pd.concat(scores, ignore_index=True)
    coefs = pd.concat(coefs, ignore_index=True)
    scores.to_csv(RESULTS / "baseline_scores.csv", index=False)
    coefs.to_csv(RESULTS / "baseline_coefficients.csv", index=False)

    print("=== 5-fold CV, cheap feature blocks only (no language model)")
    print(
        scores.pivot_table(index=["dataset", "target"], columns=["features", "split"],
                           values="mean").round(3).to_string()
    )
    print("\n=== strongest standardised coefficients in the `cheap` block")
    for (d, t), grp in coefs.groupby(["dataset", "target"], sort=False):
        top = ", ".join(f"{r.feature} ({r.coefficient:+.2f})" for r in grp.head(5).itertuples())
        print(f"  {t:<15s} {top}")


if __name__ == "__main__":
    main()
