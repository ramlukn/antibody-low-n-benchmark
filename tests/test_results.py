"""Regression tests over the committed result tables.

``results/`` is checked in so the README renders without anyone running the
sweep.  That makes the tables part of the published artefact, and every headline
number in the README is read off them.  These tests pin those numbers, so a
re-run that moves a claim has to update the prose deliberately instead of
silently disagreeing with it.

They also check the *structure* of the 23,040-row sweep: a balanced design is
what the paired statistics assume, and an unbalanced one would quietly bias
every mean in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lown.config import RESULTS

# Tolerance for a number quoted to three decimals in the prose.
QUOTED = 5e-4


def load(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not committed; run `make all`")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def sweep():
    return load("learning_curves.csv")


# ------------------------------------------------- structure of the sweep ----
def test_sweep_row_count_matches_the_readme(sweep):
    """The README opens with "23,040 fits"."""
    assert len(sweep) == 23_040


def test_sweep_grid_is_the_advertised_one(sweep):
    assert sorted(sweep["features"].unique()) == [
        "aac", "biophys", "cheap", "esm2", "esm2+cheap", "random",
    ]
    assert sorted(sweep["head"].unique()) == ["gbm", "linear", "mlp"]
    assert sorted(sweep["split"].unique()) == ["cluster", "random"]
    assert sweep["seed"].nunique() == 20
    assert sorted(sweep["dataset"].unique()) == ["SAbDab_Chen", "TAP"]


def test_sweep_design_is_perfectly_balanced(sweep):
    """Every cell must hold all 6 feature sets x 3 heads.

    An unbalanced design would make the seed-paired Wilcoxon tests compare
    unequal sets and bias every aggregated mean.
    """
    sizes = sweep.groupby(
        ["dataset", "target", "split", "seed", "n_request"]
    ).size().unique()
    assert sizes.tolist() == [18]


def test_no_degenerate_cells_survived_the_full_sweep(sweep):
    """At N=25 with a 20% positive rate this was a real risk; none materialised."""
    assert not sweep["degenerate"].any()


def test_every_classification_row_has_a_finite_auc(sweep):
    clf = sweep[sweep["task"] == "classification"]
    assert len(clf) > 0
    assert np.isfinite(clf["auc"]).all()


def test_every_regression_row_has_a_finite_spearman(sweep):
    reg = sweep[sweep["task"] == "regression"]
    assert len(reg) > 0
    assert np.isfinite(reg["spearman"]).all()


def test_scores_are_within_their_metric_bounds(sweep):
    clf = sweep[sweep["task"] == "classification"]
    assert clf["auc"].between(0.0, 1.0).all()
    assert clf["ap"].between(0.0, 1.0).all()
    reg = sweep[sweep["task"] == "regression"]
    assert reg["spearman"].between(-1.0, 1.0).all()


def test_test_set_is_fixed_within_a_seed(sweep):
    """The protocol claim, verified on the committed output rather than in code."""
    varying = sweep.groupby(["dataset", "target", "split", "seed"])["n_test"].nunique()
    assert (varying == 1).all()


def test_feature_sets_share_identical_cells_within_a_seed(sweep):
    """The paired design, verified on the committed output."""
    key = ["dataset", "target", "split", "seed", "head", "n_request"]
    counts = sweep.groupby(key)["features"].nunique()
    assert (counts == 6).all()


def test_training_sizes_never_exceed_the_pool(sweep):
    assert (sweep["n_train"] > 0).all()
    # The "all" point uses the whole pool, which is the complement of the test set.
    for (ds, tgt, sp, seed), g in sweep.groupby(["dataset", "target", "split", "seed"]):
        pool = g["n_train"].max()
        assert (g["n_train"] <= pool).all()


# ------------------------------------------------------ the leakage claim ----
def test_committed_leakage_summary_matches_the_readme():
    """"65% of a random SAbDab_Chen test set has a near-duplicate in training",
    and the cluster split has exactly none."""
    leak = load("leakage_summary.csv").set_index(["dataset", "split"])
    col = "test_with_near_duplicate_in_train"

    assert leak.loc[("SAbDab_Chen", "random"), col] == pytest.approx(0.654, abs=0.001)
    assert leak.loc[("SAbDab_Chen", "cluster"), col] == 0.0
    assert leak.loc[("TAP", "cluster"), col] == 0.0
    # TAP is 241 distinct therapeutics, so even a random split barely leaks.
    assert leak.loc[("TAP", "random"), col] < 0.10


def test_cluster_splits_leak_nothing_for_either_dataset():
    leak = load("leakage_summary.csv")
    assert (leak.loc[leak["split"] == "cluster", "test_with_near_duplicate_in_train"] == 0).all()


def test_conserved_framework_collapse_matches_the_readme():
    """"Clustering the whole Fv at 80% identity puts 1,173 of 2,409 into one cluster".

    This is the observation that motivates the union criterion, so it is worth
    pinning: a naive CD-HIT threshold is meaningless on antibodies.
    """
    clusters = load("cluster_summary.csv").set_index(["dataset", "criterion"])
    naive = clusters.loc[("SAbDab_Chen", "fv @ 80%")]
    assert naive["n_items"] == 2409
    assert naive["largest_cluster"] == 1173

    union = clusters.loc[("SAbDab_Chen", "union (default)")]
    assert union["n_clusters"] == 971
    # The union has to be strictly finer than the collapsed naive threshold and
    # strictly coarser than either criterion on its own would suggest.
    assert union["largest_cluster"] < naive["largest_cluster"]
    assert union["n_clusters"] <= clusters.loc[("SAbDab_Chen", "fv @ 90%"), "n_clusters"]
    assert union["n_clusters"] <= clusters.loc[("SAbDab_Chen", "cdr3 @ 80%"), "n_clusters"]


def test_tap_has_almost_no_near_duplicates():
    """The mechanism behind TAP's ~zero leakage tax."""
    clusters = load("cluster_summary.csv").set_index(["dataset", "criterion"])
    union = clusters.loc[("TAP", "union (default)")]
    assert union["n_clusters"] == 219
    assert union["n_clusters"] / union["n_items"] > 0.9


# --------------------------------------------------------- the honesty tax ---
def test_split_gap_matches_the_readme_table():
    gap = load("split_gap.csv").set_index(["dataset", "target"])

    chen = gap.loc[("SAbDab_Chen", "developability")]
    assert chen["random_"] == pytest.approx(0.894, abs=0.001)
    assert chen["cluster"] == pytest.approx(0.799, abs=0.001)
    assert chen["gap"] == pytest.approx(0.096, abs=0.001)

    expected = {"PPC": 0.046, "CDR_Length": 0.001, "SFvCSP": -0.008,
                "PSH": -0.013, "PNC": -0.036}
    for target, value in expected.items():
        assert gap.loc[("TAP", target), "gap"] == pytest.approx(value, abs=0.001)


def test_the_split_gap_is_larger_than_any_feature_set_gap(sweep):
    """The README's structural claim: the split matters more than the features.

    The SAbDab split gap is 0.096.  The spread between the best and worst
    *informative* feature sets (excluding the random control) must be smaller
    than that at every training-set size.
    """
    gap = load("split_gap.csv").set_index(["dataset", "target"])
    split_gap = gap.loc[("SAbDab_Chen", "developability"), "gap"]

    chen = sweep[
        (sweep["dataset"] == "SAbDab_Chen")
        & (sweep["split"] == "cluster")
        & (sweep["features"] != "random")
    ]
    per_n = chen.groupby(["n_request", "features"])["auc"].mean().unstack()
    spread = (per_n.max(axis=1) - per_n.min(axis=1)).max()
    assert spread < split_gap, f"feature spread {spread:.3f} >= split gap {split_gap:.3f}"


def test_tap_cdr_length_is_the_giveaway_target(sweep):
    """"One of the five headline TAP targets is a giveaway."

    CDR_Length is near-solved at the smallest N by every informative feature
    set, because chain length alone correlates with it at rho ~= 0.996.  A
    benchmark reporting a mean over all five TAP metrics without saying so is
    flattering itself.

    Stated as dominance rather than an absolute threshold: at N=25 under the
    linear head, CDR_Length must beat all four other TAP targets for *every*
    informative feature set.  (The MLP is genuinely untrained at N=25 and scores
    near zero on everything, so a mean across heads would hide this entirely.)
    """
    at_25 = sweep[
        (sweep["dataset"] == "TAP")
        & (sweep["n_request"] == "25")
        & (sweep["head"] == "linear")
        & (sweep["features"] != "random")
    ]
    per_target = at_25.groupby(["features", "target"])["spearman"].mean().unstack()
    others = per_target.drop(columns="CDR_Length")
    assert (per_target["CDR_Length"] > others.max(axis=1)).all(), (
        f"CDR_Length is not uniformly the easiest target:\n{per_target.round(3)}"
    )
    # And it is already high in absolute terms, for every feature set.
    assert per_target["CDR_Length"].min() > 0.65


def test_tap_cdr_length_is_essentially_solved_by_the_best_head(sweep):
    """The best-of-head envelope the README's figure shows."""
    at_25 = sweep[
        (sweep["dataset"] == "TAP")
        & (sweep["n_request"] == "25")
        & (sweep["features"] != "random")
        & (sweep["target"] == "CDR_Length")
    ]
    envelope = at_25.groupby(["features", "head"])["spearman"].mean().groupby("features").max()
    assert envelope.min() > 0.65
    assert envelope.max() > 0.80


# ------------------------------------------------- ESM-2 vs cheap features ---
def test_paired_comparison_matches_the_readme_numbers():
    """The headline table: where the language model actually helps."""
    paired = load("paired_esm_vs_cheap.csv")
    paired = paired.set_index(["dataset", "target", "split", "head", "n_request"])

    at_25 = paired.loc[("SAbDab_Chen", "developability", "cluster", "linear", "25")]
    assert at_25["mean_delta"] == pytest.approx(-0.066, abs=0.001)
    assert at_25["p_wilcoxon"] < 1e-4

    at_100 = paired.loc[("SAbDab_Chen", "developability", "cluster", "linear", "100")]
    assert at_100["mean_delta"] == pytest.approx(-0.053, abs=0.001)

    # The one clear win, worth naming precisely rather than rounding up.
    mlp_all = paired.loc[("SAbDab_Chen", "developability", "cluster", "mlp", "all")]
    assert mlp_all["mean_delta"] == pytest.approx(0.026, abs=0.001)
    assert mlp_all["p_wilcoxon"] == pytest.approx(0.0009, abs=0.0002)


def test_every_paired_row_uses_all_twenty_seeds():
    paired = load("paired_esm_vs_cheap.csv")
    assert (paired["n_seeds"] == 20).all()


def test_esm_loses_at_low_n_on_developability():
    """The central negative result, stated as a direction rather than a value."""
    paired = load("paired_esm_vs_cheap.csv")
    low_n = paired[
        (paired["target"] == "developability")
        & (paired["head"] == "linear")
        & (paired["n_request"].isin(["25", "50", "100"]))
    ]
    assert len(low_n) > 0
    assert (low_n["mean_delta"] < 0).all()


def test_cheap_descriptors_win_outright_on_tap():
    """"On all five TAP regression targets, the cheap descriptors win outright."

    Checked under the linear head, where the README makes the claim: no TAP
    target shows a significant ESM-2 advantage at the largest N.
    """
    paired = load("paired_esm_vs_cheap.csv")
    tap = paired[
        (paired["dataset"] == "TAP")
        & (paired["head"] == "linear")
        & (paired["n_request"] == "all")
    ]
    assert len(tap) == 10  # five targets x two split types
    significant_wins = tap[(tap["mean_delta"] > 0) & (tap["p_wilcoxon"] < 0.05)]
    assert significant_wins.empty, f"unexpected ESM-2 win:\n{significant_wins}"


def test_charge_symmetry_is_where_descriptors_win_biggest():
    """SFvCSP is the target the cheap set nearly encodes by construction."""
    paired = load("paired_esm_vs_cheap.csv")
    linear_all = paired[
        (paired["dataset"] == "TAP")
        & (paired["head"] == "linear")
        & (paired["split"] == "cluster")
        & (paired["n_request"] == "all")
    ].set_index("target")
    assert linear_all.loc["SFvCSP", "mean_delta"] == pytest.approx(-0.136, abs=0.001)
    assert linear_all["mean_delta"].idxmin() == "SFvCSP"


# ------------------------------------------------------- the random floor ----
def test_random_features_are_the_floor(sweep):
    """The information-free control must not beat an informative feature set.

    If noise ever won, the fitting procedure would be manufacturing skill and
    every other number in the README would be suspect.
    """
    for (ds, tgt, sp, head), g in sweep.groupby(["dataset", "target", "split", "head"]):
        metric = "auc" if g["task"].iloc[0] == "classification" else "spearman"
        means = g.groupby("features")[metric].mean()
        assert means["random"] <= means.drop("random").max() + 1e-9, (
            f"noise beat every real feature set on {ds}/{tgt}/{sp}/{head}"
        )


def test_random_features_score_near_chance_on_classification(sweep):
    chance = sweep[
        (sweep["task"] == "classification") & (sweep["features"] == "random")
    ]["auc"].mean()
    assert 0.40 < chance < 0.60, f"noise scored {chance:.3f} AUC"


# -------------------------------------------------------- the headline row ---
def test_headline_is_consistent_with_the_split_gap_table():
    headline = load("headline.csv")
    gap = load("split_gap.csv").set_index(["dataset", "target"])
    chen = headline[
        (headline["dataset"] == "SAbDab_Chen") & (headline["split"] == "cluster")
    ]
    assert chen["mean"].max() == pytest.approx(
        gap.loc[("SAbDab_Chen", "developability"), "cluster"], abs=1e-9
    )


def test_extrapolation_table_flags_its_own_reliability():
    """The README leans on "reaching 0.85 needs tens of thousands" being unreliable."""
    extra = load("scaling_extrapolation.csv")
    assert "reliable" in extra.columns
    assert "max_extrapolation_factor" in extra.columns
    # Anything past 10x the measured N must not be marked reliable.
    flagged = extra.dropna(subset=["max_extrapolation_factor"])
    overreach = flagged[flagged["max_extrapolation_factor"] > 10.0]
    assert not overreach["reliable"].any()
