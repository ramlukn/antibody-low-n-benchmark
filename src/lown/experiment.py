"""Phase 3/4 -- the learning-curve experiment.

For each dataset x target x split mode x feature set x head x training-set
size N x seed:

  1. partition into a train pool and a test set (random or cluster-held-out),
  2. draw a *nested* subsample of size N from the train pool, so that the
     N=25 set is a subset of the N=50 set for the same seed and the curve is
     paired rather than independent at every point,
  3. fit, predict on the held-out test set, score.

The test set is fixed per seed and never subsampled, so every point on a
curve is evaluated on exactly the same antibodies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import (
    FEATURE_SETS,
    HEADS,
    N_GRID_CHEN,
    N_GRID_TAP,
    N_SEEDS,
    TEST_FRACTION,
)
from .features import build_matrix, random_features
from .heads import fit_predict, score
from .splits import identity_clusters, make_splits


@dataclass
class Task:
    dataset: str
    target: str
    task: str  # "classification" | "regression"
    n_grid: list = field(default_factory=list)


def chen_task() -> Task:
    from .config import CHEN

    return Task(CHEN, "developability", "classification", list(N_GRID_CHEN))


def tap_tasks() -> list[Task]:
    from .config import TAP, TAP_TARGETS

    return [Task(TAP, t, "regression", list(N_GRID_TAP)) for t in TAP_TARGETS]


def feature_bank(df, esm_lookup, sets=FEATURE_SETS) -> dict[str, np.ndarray]:
    """Materialise every requested feature matrix once, up front."""
    from .embeddings import antibody_matrix

    bank: dict[str, np.ndarray] = {}
    cheap = None
    esm = None
    for name in sets:
        if name == "random":
            bank[name] = random_features(df, dim=64, seed=0)
        elif name in ("aac", "biophys", "cdr"):
            bank[name] = build_matrix(df, name)
        elif name == "cheap":
            cheap = cheap if cheap is not None else build_matrix(df, "cheap")
            bank[name] = cheap
        elif name == "esm2":
            esm = esm if esm is not None else antibody_matrix(df, esm_lookup)
            bank[name] = esm
        elif name == "esm2+cheap":
            esm = esm if esm is not None else antibody_matrix(df, esm_lookup)
            cheap = cheap if cheap is not None else build_matrix(df, "cheap")
            bank[name] = np.hstack([esm, cheap])
        else:
            raise KeyError(name)
    return bank


def _resolve_grid(grid, pool_size: int) -> list[tuple[str, int]]:
    """(requested label, actual size) pairs.

    Cluster-held-out splits partition whole clusters, so the training pool is
    not exactly the same size for every seed.  Keeping the requested label
    alongside the realised size lets the "all" point be averaged across seeds
    instead of scattering into one column per pool size.
    """
    out: list[tuple[str, int]] = []
    seen: set[int] = set()
    for n in grid:
        size = pool_size if n == "all" else int(n)
        if size <= pool_size and size not in seen:
            out.append((str(n), size))
            seen.add(size)
    if pool_size not in seen:
        out.append(("all", pool_size))
    return sorted(out, key=lambda t: t[1])


def _one_cell(X, y, task, feat_name, head, split, grid, seed):
    """All training-set sizes for one (feature set, head, seed) combination."""
    train_pool, test_idx = split
    rng = np.random.default_rng(1000 + seed)
    order = rng.permutation(train_pool)
    X_te, y_te = X[test_idx], y[test_idx]
    rows = []
    for label, n in _resolve_grid(grid, len(train_pool)):
        sub = order[:n]
        y_tr = y[sub]
        base = dict(
            features=feat_name, head=head, n_request=label, n_train=n,
            seed=seed, n_test=len(test_idx),
        )
        if task == "classification" and len(np.unique(y_tr)) < 2:
            rows.append({**base, "auc": np.nan, "ap": np.nan, "degenerate": True})
            continue
        try:
            pred = fit_predict(head, task, X[sub], y_tr, X_te, seed)
        except Exception as exc:  # one pathological cell must not kill a long sweep
            rows.append({**base, "degenerate": True, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append({**base, **score(task, y_te, pred), "degenerate": False})
    return rows


def run_task(
    task: Task,
    df: pd.DataFrame,
    bank: dict[str, np.ndarray],
    split_mode: str,
    groups: np.ndarray | None = None,
    n_seeds: int = N_SEEDS,
    heads: list[str] = HEADS,
    n_jobs: int = -1,
    verbose: int = 5,
) -> pd.DataFrame:
    y = df[task.target if task.target in df.columns else "y"].to_numpy()
    stratify = y if task.task == "classification" else None
    splits = make_splits(
        len(df), n_seeds, TEST_FRACTION, split_mode, groups=groups, stratify=stratify
    )
    seeds = list(range(n_seeds))

    # One job per (feature set, head, seed): fine-grained enough that the
    # expensive cells (ESM features + gradient boosting at large N) spread
    # across cores instead of serialising behind one another.
    jobs = [
        delayed(_one_cell)(bank[f], y, task.task, f, h, splits[s], task.n_grid, s)
        for f in bank
        for h in heads
        for s in seeds
    ]
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)

    rows = [r for chunk in out for r in chunk]
    res = pd.DataFrame(rows)
    res.insert(0, "split", split_mode)
    res.insert(0, "target", task.target)
    res.insert(0, "dataset", task.dataset)
    res.insert(3, "task", task.task)
    return res


def cluster_groups(df) -> np.ndarray:
    return identity_clusters(df)
