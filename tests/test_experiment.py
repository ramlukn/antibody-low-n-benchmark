"""The protocol invariants: nested subsampling and a fixed test set.

The README claims each learning curve is *paired* -- the N=25 training set is a
subset of the N=50 set, and every point on a curve is scored on exactly the
same test antibodies.  Both claims are what licenses the Wilcoxon signed-rank
tests and the one-standard-deviation bands, so both are tested by observing
which rows actually reach the fit.

The trick throughout: build a feature matrix where row *i* is literally
``[i]``.  A recording stub in place of ``fit_predict`` then reports the exact
indices the protocol handed it.
"""
from __future__ import annotations

import numpy as np
import pytest

from lown.experiment import Task, _one_cell, _resolve_grid, run_task
from lown.splits import identity_clusters, make_splits


def identity_matrix(n: int) -> np.ndarray:
    """Row i is [i], so a recorded X_tr reveals the indices used."""
    return np.arange(n, dtype=float).reshape(-1, 1)


@pytest.fixture
def recorder(monkeypatch):
    """Replace the fit with a stub that records the indices it was given."""
    calls = []

    def fake_fit_predict(head, task, X_tr, y_tr, X_te, seed):
        calls.append(
            dict(
                head=head,
                train=X_tr.ravel().astype(int).tolist(),
                test=X_te.ravel().astype(int).tolist(),
            )
        )
        rng = np.random.default_rng(len(calls))
        return rng.random(len(X_te))

    monkeypatch.setattr("lown.experiment.fit_predict", fake_fit_predict)
    return calls


# ------------------------------------------------------- nested subsampling --
def test_subsamples_are_nested(recorder):
    """N=25 must be a subset of N=50, for the same seed.

    Independent draws at each N would make the curve unpaired and the
    one-sigma bands an overstatement of the noise between adjacent points.
    """
    n = 200
    X = identity_matrix(n)
    y = np.arange(n) % 2
    train_pool, test_idx = make_splits(n, 1, 0.25, mode="random")[0]

    _one_cell(X, y, "classification", "f", "linear", (train_pool, test_idx),
              [10, 25, 50, 100, "all"], seed=3)

    train_sets = [c["train"] for c in recorder]
    assert len(train_sets) >= 4
    for smaller, larger in zip(train_sets, train_sets[1:]):
        assert len(smaller) < len(larger)
        assert set(smaller) <= set(larger), "subsample is not nested"
        # Nested *and order-preserving*: the smaller set is a prefix.
        assert larger[: len(smaller)] == smaller


def test_training_subsamples_never_touch_the_test_set(recorder):
    n = 200
    X = identity_matrix(n)
    y = np.arange(n) % 2
    train_pool, test_idx = make_splits(n, 1, 0.25, mode="random")[0]

    _one_cell(X, y, "classification", "f", "linear", (train_pool, test_idx),
              [10, 50, "all"], seed=1)

    test_set = set(test_idx.tolist())
    for call in recorder:
        assert not set(call["train"]) & test_set


def test_test_set_is_identical_at_every_point_on_the_curve(recorder):
    """Every point on a curve is scored on exactly the same antibodies."""
    n = 200
    X = identity_matrix(n)
    y = np.arange(n) % 2
    train_pool, test_idx = make_splits(n, 1, 0.25, mode="random")[0]

    _one_cell(X, y, "classification", "f", "linear", (train_pool, test_idx),
              [10, 25, 50, "all"], seed=7)

    test_sets = {tuple(c["test"]) for c in recorder}
    assert len(test_sets) == 1, "the test set moved between training-set sizes"
    assert sorted(next(iter(test_sets))) == sorted(test_idx.tolist())


def test_subsample_is_reproducible_for_a_seed(recorder):
    n = 120
    X = identity_matrix(n)
    y = np.arange(n) % 2
    split = make_splits(n, 1, 0.25, mode="random")[0]

    _one_cell(X, y, "classification", "f", "linear", split, [10, 40], seed=5)
    first = [c["train"] for c in recorder]
    recorder.clear()
    _one_cell(X, y, "classification", "f", "linear", split, [10, 40], seed=5)
    assert [c["train"] for c in recorder] == first


def test_different_seeds_draw_different_subsamples(recorder):
    n = 120
    X = identity_matrix(n)
    y = np.arange(n) % 2
    split = make_splits(n, 1, 0.25, mode="random")[0]

    _one_cell(X, y, "classification", "f", "linear", split, [40], seed=1)
    _one_cell(X, y, "classification", "f", "linear", split, [40], seed=2)
    assert recorder[0]["train"] != recorder[1]["train"]


def test_feature_sets_and_heads_share_a_subsample_within_a_seed(recorder):
    """What makes the paired Wilcoxon test legitimate.

    Comparing ESM-2 against cheap descriptors within a seed is only paired if
    both saw the same training rows and the same test rows.
    """
    n = 150
    y = np.arange(n) % 2
    split = make_splits(n, 1, 0.25, mode="random")[0]

    for feat in ("cheap", "esm2"):
        for head in ("linear", "mlp"):
            _one_cell(identity_matrix(n), y, "classification", feat, head,
                      split, [50], seed=4)

    # Each cell yields two points (N=50 and the full pool), so compare within
    # a training-set size: every feature/head combination must see the same rows.
    by_size: dict[int, set] = {}
    for call in recorder:
        by_size.setdefault(len(call["train"]), set()).add(tuple(call["train"]))
    assert by_size, "nothing was recorded"
    for size, variants in by_size.items():
        assert len(variants) == 1, f"N={size} differed between feature sets or heads"
    assert len({tuple(c["test"]) for c in recorder}) == 1


# ------------------------------------------------------- grid resolution -----
def test_resolve_grid_drops_sizes_larger_than_the_pool():
    got = _resolve_grid([25, 50, 100, 200, 500, "all"], pool_size=120)
    assert got == [("25", 25), ("50", 50), ("100", 100), ("all", 120)]


def test_resolve_grid_is_sorted_and_deduplicated():
    got = _resolve_grid([100, 25, 25, 50, "all"], pool_size=300)
    sizes = [n for _, n in got]
    assert sizes == sorted(sizes)
    assert len(sizes) == len(set(sizes))


def test_resolve_grid_always_reaches_the_whole_pool():
    """Every curve must have a point that uses the entire training pool.

    Cluster-held-out pools differ in size between seeds, so this cannot be
    assumed from the requested grid.
    """
    for pool in (37, 120, 180, 1806):
        got = _resolve_grid([25, 50, 100, 200, 500, 1000, "all"], pool_size=pool)
        assert max(n for _, n in got) == pool


def test_resolve_grid_labels_the_full_pool_all_unless_the_grid_names_it():
    """Documents a real labelling quirk.

    When the pool size coincides exactly with a requested size, that size wins
    the label and no row is tagged ``"all"``.  Aggregation keys on the realised
    ``n_train`` rather than the label, so this is cosmetic -- but it is
    surprising enough to pin, and it is why plotting code must not assume an
    ``"all"`` row exists.
    """
    assert _resolve_grid([25, 50, "all"], pool_size=50) == [("25", 25), ("50", 50)]
    assert ("all", 60) in _resolve_grid([25, 50, "all"], pool_size=60)


def test_resolve_grid_handles_a_pool_smaller_than_every_requested_size():
    assert _resolve_grid([25, 50, "all"], pool_size=8) == [("all", 8)]


# ------------------------------------------------- degenerate-cell handling --
def test_single_class_subsample_is_recorded_not_crashed():
    """A real failure mode at N=25 with a 20% positive rate.

    The cell must be reported as degenerate with a NaN score, so it is visibly
    missing from the curve rather than silently dropped or crashing an
    hour-long sweep.
    """
    n = 100
    X = identity_matrix(n)
    y = np.zeros(n, dtype=int)
    y[:2] = 1  # almost every subsample lands single-class
    train_pool, test_idx = make_splits(n, 1, 0.25, mode="random")[0]

    rows = _one_cell(X, y, "classification", "f", "linear",
                     (train_pool, test_idx), [10, 20], seed=0)
    degenerate = [r for r in rows if r["degenerate"]]
    assert degenerate, "expected at least one degenerate cell"
    for row in degenerate:
        assert np.isnan(row["auc"])
        assert row["n_train"] in (10, 20)


def test_a_failing_fit_is_caught_and_reported(monkeypatch):
    """One pathological cell must not kill a long sweep."""

    def boom(*a, **k):
        raise RuntimeError("singular matrix")

    monkeypatch.setattr("lown.experiment.fit_predict", boom)
    n = 80
    y = np.arange(n) % 2
    split = make_splits(n, 1, 0.25, mode="random")[0]

    rows = _one_cell(identity_matrix(n), y, "classification", "f", "linear",
                     split, [20, 40], seed=0)
    assert rows and all(r["degenerate"] for r in rows)
    assert all("singular matrix" in r["error"] for r in rows)


def test_regression_cells_report_regression_metrics(recorder):
    n = 120
    X = identity_matrix(n)
    y = np.linspace(0, 1, n)
    split = make_splits(n, 1, 0.25, mode="random")[0]

    rows = _one_cell(X, y, "regression", "f", "linear", split, [30, "all"], seed=0)
    for row in rows:
        assert not row["degenerate"]
        assert {"spearman", "pearson", "r2", "rmse"} <= set(row)


# --------------------------------------------------------------- run_task ----
def test_run_task_emits_one_row_per_cell_with_provenance(clonal, monkeypatch):
    """The sweep output must carry enough columns to regroup it later."""

    def fake_fit_predict(head, task, X_tr, y_tr, X_te, seed):
        rng = np.random.default_rng(seed)
        return rng.random(len(X_te))

    monkeypatch.setattr("lown.experiment.fit_predict", fake_fit_predict)

    task = Task("SyntheticPanel", "y", "classification", [10, 20, "all"])
    bank = {"aac": identity_matrix(len(clonal)), "cheap": identity_matrix(len(clonal))}
    groups = identity_clusters(clonal)

    res = run_task(task, clonal, bank, "cluster", groups=groups,
                   n_seeds=3, heads=["linear"], n_jobs=1, verbose=0)

    assert {"dataset", "target", "task", "split", "features", "head",
            "n_request", "n_train", "seed", "n_test"} <= set(res.columns)
    assert set(res["features"]) == {"aac", "cheap"}
    assert set(res["seed"]) == {0, 1, 2}
    assert (res["split"] == "cluster").all()
    assert (res["dataset"] == "SyntheticPanel").all()
    # Both feature sets saw identical (seed, n_train) cells: the paired design.
    cells = res.groupby("features").apply(
        lambda g: sorted(zip(g.seed, g.n_train)), include_groups=False
    )
    assert cells["aac"] == cells["cheap"]


def test_run_task_test_set_size_is_constant_within_a_seed(clonal, monkeypatch):
    def fake_fit_predict(head, task, X_tr, y_tr, X_te, seed):
        return np.random.default_rng(0).random(len(X_te))

    monkeypatch.setattr("lown.experiment.fit_predict", fake_fit_predict)

    task = Task("SyntheticPanel", "y", "classification", [10, 20, "all"])
    bank = {"aac": identity_matrix(len(clonal))}
    res = run_task(task, clonal, bank, "random", n_seeds=4,
                   heads=["linear"], n_jobs=1, verbose=0)

    for seed, group in res.groupby("seed"):
        assert group["n_test"].nunique() == 1
