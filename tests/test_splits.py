"""The honesty invariants.

The README's central claim is that a cluster-held-out split has no
near-duplicate leakage and a random split has plenty.  That is the load-bearing
methodological statement of the whole benchmark, so it gets tested directly
rather than inferred from the summary tables.
"""
from __future__ import annotations

import numpy as np
import pytest

from lown.splits import (
    DEFAULT_CRITERIA,
    cluster_report,
    identity_clusters,
    leakage_summary,
    make_splits,
)


# ------------------------------------------------------------- clustering ---
def test_clusters_recover_clonal_families(clonal):
    """Single-linkage clustering must not break a clonal family apart.

    A family shares a CDR3 and differs by three framework point mutations, so
    every member links to every other under either criterion.  If clustering
    split a family, the cluster-held-out split would leak.
    """
    labels = identity_clusters(clonal)
    for family, group in clonal.groupby("family").groups.items():
        assert len({labels[i] for i in group}) == 1, f"family {family} was split"


def test_clusters_do_not_merge_unrelated_families(clonal):
    labels = identity_clusters(clonal)
    by_cluster = {}
    for fam, lab in zip(clonal["family"], labels):
        by_cluster.setdefault(lab, set()).add(fam)
    merged = {k: v for k, v in by_cluster.items() if len(v) > 1}
    assert not merged, f"unrelated families merged: {merged}"


def test_distinct_panel_is_all_singletons(distinct):
    """A panel with no near-duplicates should cluster to one antibody each.

    This is the TAP-like regime, and it is why TAP's leakage tax is ~zero.
    """
    report = cluster_report(identity_clusters(distinct))
    assert report["n_clusters"] == len(distinct)
    assert report["singletons"] == len(distinct)


def test_cluster_report_is_self_consistent(clonal):
    labels = identity_clusters(clonal)
    report = cluster_report(labels)
    assert report["n_items"] == len(clonal)
    assert 1 <= report["largest_cluster"] <= report["n_items"]
    assert report["largest_frac"] == pytest.approx(
        report["largest_cluster"] / report["n_items"]
    )


def test_clustering_is_deterministic(clonal):
    a = identity_clusters(clonal)
    b = identity_clusters(clonal)
    assert np.array_equal(a, b)


def test_union_criteria_cluster_at_least_as_coarsely_as_either_alone(clonal):
    """The union of two identity rules can only merge, never split.

    The default criteria are a deliberate over-merge: a split that survives
    them is hard to accuse of leakage.  If the union ever produced *more*
    clusters than a single criterion, the conservatism argument would be void.
    """
    union = identity_clusters(clonal, criteria=DEFAULT_CRITERIA)
    for single in DEFAULT_CRITERIA:
        alone = identity_clusters(clonal, criteria=(single,))
        n_union = len(np.unique(union))
        n_alone = len(np.unique(alone))
        assert n_union <= n_alone, f"union split what {single} kept together"


def test_unknown_representation_and_mode_are_rejected(clonal):
    with pytest.raises(KeyError):
        identity_clusters(clonal, criteria=(("structure", 0.9),))
    with pytest.raises(KeyError):
        make_splits(len(clonal), 2, 0.25, mode="stratified-by-vibes")


# ------------------------------------------------------- the leakage claim --
def test_cluster_split_has_zero_near_duplicate_leakage(clonal):
    """The headline methodological claim, asserted directly."""
    groups = identity_clusters(clonal)
    splits = make_splits(len(clonal), 5, 0.25, mode="cluster", groups=groups)
    for train_idx, test_idx in splits:
        leak = leakage_summary(clonal, train_idx, test_idx, how="cdr3")
        assert leak == 0.0, f"cluster split leaked {leak:.1%} of the test set"


def test_random_split_leaks_heavily_on_a_clonal_panel(clonal):
    """The contrast that makes the honesty tax real rather than rhetorical.

    With five-member families and a 25% test set, nearly every test antibody
    should have a relative left behind in training.
    """
    splits = make_splits(len(clonal), 5, 0.25, mode="random")
    leaks = [leakage_summary(clonal, tr, te, how="cdr3") for tr, te in splits]
    assert np.mean(leaks) > 0.8, f"expected heavy leakage, got {np.mean(leaks):.1%}"


def test_cluster_split_leaks_strictly_less_than_random(clonal):
    groups = identity_clusters(clonal)
    rand = make_splits(len(clonal), 5, 0.25, mode="random")
    clus = make_splits(len(clonal), 5, 0.25, mode="cluster", groups=groups)
    rand_leak = np.mean([leakage_summary(clonal, tr, te) for tr, te in rand])
    clus_leak = np.mean([leakage_summary(clonal, tr, te) for tr, te in clus])
    assert clus_leak < rand_leak


def test_no_near_duplicate_free_panel_shows_a_leakage_gap(distinct):
    """On a panel with no near-duplicates, the split type should not matter.

    This is the mechanism behind "the leakage tax is a property of the dataset,
    not a constant".
    """
    groups = identity_clusters(distinct)
    rand = make_splits(len(distinct), 5, 0.25, mode="random")
    clus = make_splits(len(distinct), 5, 0.25, mode="cluster", groups=groups)
    rand_leak = np.mean([leakage_summary(distinct, tr, te) for tr, te in rand])
    clus_leak = np.mean([leakage_summary(distinct, tr, te) for tr, te in clus])
    assert rand_leak == pytest.approx(0.0)
    assert clus_leak == pytest.approx(0.0)


# ---------------------------------------------------------- split mechanics --
def test_splits_partition_without_overlap(clonal):
    groups = identity_clusters(clonal)
    for mode, kw in [("random", {}), ("cluster", {"groups": groups})]:
        for train_idx, test_idx in make_splits(len(clonal), 4, 0.25, mode=mode, **kw):
            assert not set(train_idx) & set(test_idx), f"{mode}: train/test overlap"
            assert len(set(train_idx) | set(test_idx)) == len(clonal), f"{mode}: lost rows"


def test_splits_are_reproducible_across_calls(clonal):
    """Seed pairing depends on this.

    The sweep calls ``make_splits`` once per (dataset, target, split mode) and
    indexes it by seed, and the Wilcoxon tests pair feature sets *within* a
    seed.  If two calls with the same arguments disagreed, every paired test in
    the README would be comparing different test sets.
    """
    groups = identity_clusters(clonal)
    for mode, kw in [("random", {}), ("cluster", {"groups": groups})]:
        a = make_splits(len(clonal), 4, 0.25, mode=mode, **kw)
        b = make_splits(len(clonal), 4, 0.25, mode=mode, **kw)
        for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
            assert np.array_equal(tr_a, tr_b)
            assert np.array_equal(te_a, te_b)


def test_stratified_random_split_preserves_class_balance(clonal):
    y = clonal["y"].to_numpy()
    base = y.mean()
    for train_idx, test_idx in make_splits(
        len(clonal), 5, 0.25, mode="random", stratify=y
    ):
        assert y[test_idx].mean() == pytest.approx(base, abs=0.12)


def test_cluster_mode_requires_groups(clonal):
    with pytest.raises(ValueError, match="group labels"):
        make_splits(len(clonal), 3, 0.25, mode="cluster", groups=None)


def test_cluster_split_holds_out_whole_clusters(clonal):
    groups = identity_clusters(clonal)
    for train_idx, test_idx in make_splits(len(clonal), 5, 0.25, mode="cluster", groups=groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])


# ------------------------------------------------------------ edge cases ----
def test_failed_cdr_extraction_merges_rather_than_leaks(clonal):
    """An undocumented consequence of the regex CDR annotation, pinned.

    ``rapidfuzz.fuzz.ratio("", "") == 100``, so antibodies where *both* CDR3
    regexes fail to fire share an empty representation and get linked into one
    cluster.  On the real SAbDab panel this affects 15 of 2,409 antibodies.

    The direction is what matters: they are over-merged, so they end up on the
    same side of the partition.  That costs a little test-set size and can
    never manufacture leakage.  This test fails if that direction ever flips.
    """
    df = clonal.copy()
    # Truncate the motifs on the first two rows of two *different* families, so
    # cdr_h3 and cdr_l3 both return "".
    victims = [0, 5]
    df.loc[victims, "heavy"] = "AAAA" + "DEFGHIKLMNPQRSTVY" * 5
    df.loc[victims, "light"] = "AAAA" + "YVTSRQPNMLKIHGFED" * 5

    labels = identity_clusters(df)
    assert labels[victims[0]] == labels[victims[1]], (
        "empty-CDR3 antibodies must be over-merged, not separated"
    )

    groups = labels
    for train_idx, test_idx in make_splits(len(df), 5, 0.25, mode="cluster", groups=groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])


def test_leakage_summary_is_a_fraction(clonal):
    splits = make_splits(len(clonal), 3, 0.25, mode="random")
    for tr, te in splits:
        for how in ("fv", "cdr3"):
            leak = leakage_summary(clonal, tr, te, how=how)
            assert 0.0 <= leak <= 1.0


def test_identical_panel_collapses_to_one_cluster():
    """Sanity floor: 40 copies of one antibody is one cluster, not 40."""
    import pandas as pd

    from tests.conftest import make_heavy, make_light

    rng = np.random.default_rng(0)
    h = make_heavy("ARDYWGQGTLVTV"[:12], rng)
    l = make_light("QQSYSTPLT", rng)
    df = pd.DataFrame({"heavy": [h] * 40, "light": [l] * 40})
    assert len(np.unique(identity_clusters(df))) == 1
