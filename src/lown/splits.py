"""Random vs. cluster-held-out train/test splits.

A random split lets near-duplicate antibodies sit on both sides of the
partition.  Public antibody sets are full of these -- germline-shared light
chains, affinity-matured variants of one parent, the same therapeutic under
two names -- so a random split measures interpolation within a family, not
generalisation to a new one.

We cluster by sequence identity (single linkage over a similarity threshold,
i.e. the same idea as CD-HIT without the dependency) and hold out whole
clusters.  Two representations are offered because they behave very
differently on antibodies: the full Fv is dominated by conserved framework
and so collapses into giant clusters, while the CDR3 loops are the
hypervariable part that actually distinguishes one antibody from another.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit

from .features import cdr_h3, cdr_l3


def _representation(df, how: str) -> list[str]:
    if how == "fv":
        return [h + l for h, l in zip(df["heavy"], df["light"])]
    if how == "cdr3":
        return [cdr_h3(h) + cdr_l3(l) for h, l in zip(df["heavy"], df["light"])]
    raise KeyError(how)


DEFAULT_CRITERIA: tuple[tuple[str, float], ...] = (("fv", 0.90), ("cdr3", 0.80))


def _similar_pairs(seqs: list[str], identity: float):
    from rapidfuzz import fuzz, process

    sim = process.cdist(
        seqs, seqs, scorer=fuzz.ratio, score_cutoff=identity * 100.0,
        dtype=np.uint8, workers=-1,
    )
    return np.nonzero(sim)


def identity_clusters(df, criteria=DEFAULT_CRITERIA) -> np.ndarray:
    """Single-linkage clusters under the union of one or more identity rules.

    Default: two antibodies are linked if they are >=90% identical over the
    whole Fv *or* >=80% identical over their concatenated CDR3 loops.  The
    union is deliberately conservative -- it catches both near-duplicate
    frameworks and clonally related variants, and a split that survives it is
    hard to accuse of leakage.
    """
    n = len(df)
    rows, cols = [], []
    for how, identity in criteria:
        r, c = _similar_pairs(_representation(df, how), identity)
        rows.append(r)
        cols.append(c)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    graph = coo_matrix((np.ones(rows.size, dtype=np.uint8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def cluster_report(labels: np.ndarray) -> dict:
    _, sizes = np.unique(labels, return_counts=True)
    return {
        "n_items": int(labels.size),
        "n_clusters": int(sizes.size),
        "largest_cluster": int(sizes.max()),
        "largest_frac": float(sizes.max() / labels.size),
        "singletons": int((sizes == 1).sum()),
        "median_size": float(np.median(sizes)),
    }


def make_splits(
    n: int,
    n_seeds: int,
    test_fraction: float,
    mode: str = "random",
    groups: np.ndarray | None = None,
    stratify: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n_seeds` independent (train_pool, test) partitions."""
    x = np.zeros((n, 1))
    if mode == "random":
        if stratify is not None:
            sp = StratifiedShuffleSplit(n_seeds, test_size=test_fraction, random_state=0)
            return list(sp.split(x, stratify))
        sp = ShuffleSplit(n_seeds, test_size=test_fraction, random_state=0)
        return list(sp.split(x))
    if mode == "cluster":
        if groups is None:
            raise ValueError("cluster mode needs group labels")
        sp = GroupShuffleSplit(n_seeds, test_size=test_fraction, random_state=0)
        return list(sp.split(x, groups=groups))
    raise KeyError(mode)


def leakage_summary(df, train_idx, test_idx, how: str = "cdr3") -> float:
    """Fraction of test antibodies with a >=80%-identity neighbour in train."""
    from rapidfuzz import fuzz, process

    seqs = _representation(df, how)
    tr = [seqs[i] for i in train_idx]
    te = [seqs[i] for i in test_idx]
    sim = process.cdist(te, tr, scorer=fuzz.ratio, score_cutoff=80.0, dtype=np.uint8, workers=-1)
    return float((sim.max(axis=1) > 0).mean())
