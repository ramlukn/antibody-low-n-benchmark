"""The pooling ablation's analysis helpers.

``scripts/05_pooling.py`` turns the ablation sweep into a verdict on one
specific prediction: if mean pooling is what holds frozen ESM-2 back, max
pooling should beat it on TAP's three *patch* targets specifically.  The two
helpers that produce that verdict are worth testing on synthetic input where the
right answer is known, because on real data a sign error would just look like a
finding.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pooling():
    """Import ``scripts/05_pooling.py``, whose name is not a valid identifier."""
    spec = importlib.util.spec_from_file_location(
        "pooling_script", ROOT / "scripts" / "05_pooling.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sweep_frame(deltas: dict[str, float], n_seeds: int = 20, target="PSH",
                baseline_value: float = 0.5, noise: float = 0.01, seed: int = 0):
    """A synthetic ablation sweep where each variant sits a known delta above mean.

    ``deltas`` maps feature label -> true offset from ``esm2_mean``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seeds):
        base = baseline_value + rng.normal(0, noise)
        rows.append(dict(dataset="TAP", target=target, split="cluster", task="regression",
                         features="esm2_mean", head="linear", n_request="all", seed=s,
                         value=base))
        for label, delta in deltas.items():
            rows.append(dict(dataset="TAP", target=target, split="cluster",
                             task="regression", features=label, head="linear",
                             n_request="all", seed=s, value=base + delta))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- paired_vs ---
def test_paired_vs_recovers_a_known_delta(pooling):
    raw = sweep_frame({"esm2_max": 0.04, "cheap": 0.10})
    got = pooling.paired_vs(raw, "esm2_mean").set_index("features")

    assert got.loc["esm2_max", "mean_delta"] == pytest.approx(0.04, abs=1e-9)
    assert got.loc["cheap", "mean_delta"] == pytest.approx(0.10, abs=1e-9)
    # A constant offset across seeds is as significant as Wilcoxon can report.
    assert got.loc["esm2_max", "p_wilcoxon"] < 0.001
    assert (got["n_seeds"] == 20).all()


def test_paired_vs_reports_the_sign_correctly(pooling):
    """A sign error here would invert the headline of the ablation."""
    raw = sweep_frame({"esm2_max": -0.06})
    got = pooling.paired_vs(raw, "esm2_mean").set_index("features")
    assert got.loc["esm2_max", "mean_delta"] < 0


def test_paired_vs_excludes_the_baseline_from_its_own_comparison(pooling):
    raw = sweep_frame({"esm2_max": 0.02})
    got = pooling.paired_vs(raw, "esm2_mean")
    assert "esm2_mean" not in set(got["features"])


def test_paired_vs_skips_groups_with_too_few_seeds(pooling):
    """Quick mode runs three seeds, which is not enough for a signed-rank test."""
    raw = sweep_frame({"esm2_max": 0.05}, n_seeds=3)
    assert pooling.paired_vs(raw, "esm2_mean").empty


def test_paired_vs_returns_nothing_when_the_baseline_is_absent(pooling):
    raw = sweep_frame({"esm2_max": 0.05})
    assert pooling.paired_vs(raw, "not_a_feature").empty


def test_paired_vs_survives_identical_scores(pooling):
    """Zero differences make Wilcoxon raise; the p-value must come back NaN."""
    raw = sweep_frame({"esm2_max": 0.0}, noise=0.0)
    got = pooling.paired_vs(raw, "esm2_mean").set_index("features")
    assert got.loc["esm2_max", "mean_delta"] == pytest.approx(0.0)
    assert np.isnan(got.loc["esm2_max", "p_wilcoxon"])


def test_paired_vs_keeps_groups_separate(pooling):
    a = sweep_frame({"esm2_max": 0.05}, target="PSH")
    b = sweep_frame({"esm2_max": -0.05}, target="SFvCSP", seed=1)
    got = pooling.paired_vs(pd.concat([a, b], ignore_index=True), "esm2_mean")
    got = got.set_index("target")
    assert got.loc["PSH", "mean_delta"] > 0
    assert got.loc["SFvCSP", "mean_delta"] < 0


def test_paired_vs_labels_the_comparison_readably(pooling):
    raw = sweep_frame({"esm2_max": 0.01})
    got = pooling.paired_vs(raw, "esm2_mean")
    assert got["comparison"].iloc[0] == "esm2_max - esm2_mean"
    assert (got["baseline"] == "esm2_mean").all()


# ----------------------------------------------------------- patch_verdict ---
def _paired_rows(per_target: dict[str, float], p: float = 0.001):
    """One paired row per target, as ``patch_verdict`` expects to receive them."""
    return pd.DataFrame(
        [
            dict(dataset="TAP", target=t, split="cluster", head="linear",
                 n_request="all", comparison="esm2_max - esm2_mean",
                 features="esm2_max", baseline="esm2_mean", n_seeds=20,
                 mean_delta=d, sd_delta=0.01, p_wilcoxon=p)
            for t, d in per_target.items()
        ]
    )


def test_patch_verdict_separates_patch_from_non_patch_targets(pooling):
    paired = _paired_rows(
        {"PSH": 0.05, "PPC": 0.04, "PNC": 0.06,          # patch properties
         "SFvCSP": -0.02, "CDR_Length": -0.01}            # not
    )
    got = pooling.patch_verdict(paired).set_index("target_kind")

    assert got.loc["patch", "n_targets"] == 3
    assert got.loc["not patch", "n_targets"] == 2
    assert got.loc["patch", "mean_delta"] == pytest.approx(0.05, abs=1e-9)
    assert got.loc["not patch", "mean_delta"] == pytest.approx(-0.015, abs=1e-9)


def test_patch_verdict_would_detect_the_predicted_effect(pooling):
    """If pooling really were the problem, this is the shape the table would take."""
    paired = _paired_rows(
        {"PSH": 0.08, "PPC": 0.07, "PNC": 0.09, "SFvCSP": 0.00, "CDR_Length": -0.01}
    )
    got = pooling.patch_verdict(paired).set_index("target_kind")
    assert got.loc["patch", "mean_delta"] > got.loc["not patch", "mean_delta"]
    assert got.loc["patch", "n_significant_wins"] == 3


def test_patch_verdict_counts_only_significant_wins(pooling):
    """A positive delta with a weak p-value is not a win."""
    strong = _paired_rows({"PSH": 0.05, "PPC": 0.04, "PNC": 0.06}, p=0.001)
    weak = _paired_rows({"PSH": 0.05, "PPC": 0.04, "PNC": 0.06}, p=0.40)
    assert pooling.patch_verdict(strong)["n_significant_wins"].sum() == 3
    assert pooling.patch_verdict(weak)["n_significant_wins"].sum() == 0


def test_patch_verdict_does_not_count_significant_losses_as_wins(pooling):
    """The real result: large, significant, and negative."""
    paired = _paired_rows({"PSH": -0.05, "PPC": -0.18, "PNC": -0.15}, p=1e-5)
    got = pooling.patch_verdict(paired)
    assert got["n_significant_wins"].sum() == 0
    assert (got["mean_delta"] < 0).all()


def test_patch_verdict_reports_the_range(pooling):
    paired = _paired_rows({"PSH": -0.01, "PPC": -0.20, "PNC": -0.10})
    got = pooling.patch_verdict(paired).set_index("target_kind")
    assert got.loc["patch", "worst"] == pytest.approx(-0.20)
    assert got.loc["patch", "best"] == pytest.approx(-0.01)


def test_patch_verdict_ignores_other_readouts_and_sizes(pooling):
    """Only max-vs-mean at the largest N feeds the verdict."""
    paired = pd.concat(
        [
            _paired_rows({"PSH": -0.05}),
            _paired_rows({"PSH": 0.99}).assign(features="esm2_cls"),
            _paired_rows({"PSH": 0.99}).assign(n_request="25"),
            _paired_rows({"PSH": 0.99}).assign(baseline="cheap"),
        ],
        ignore_index=True,
    )
    got = pooling.patch_verdict(paired)
    assert len(got) == 1
    assert got["mean_delta"].iloc[0] == pytest.approx(-0.05)


def test_patch_targets_are_the_three_tap_patch_metrics(pooling):
    """Guards the claim in the docstring against a silent edit."""
    assert set(pooling.PATCH_TARGETS) == {"PSH", "PPC", "PNC"}


def test_variant_labels_cover_every_pooling(pooling):
    from lown.embeddings import POOLINGS

    assert set(pooling.VARIANTS) == set(POOLINGS)
