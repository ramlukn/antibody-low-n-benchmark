"""Power-law fit and the "how many more antibodies?" inversion.

These are the most load-bearing numbers in the README and the least
constrained by data -- seven points, three free parameters.  The tests below
check that the fit recovers a known curve, that the inversion is a true
inverse, that the deliberate ceiling cap actually binds, and that the code
refuses to fit rather than inventing a number when it cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lown.scaling import (
    CEILING_HEADROOM,
    MIN_POINTS,
    extrapolate,
    fit_curve,
    n_for_target,
    power_law,
)


def synthetic_curve(ceiling=0.85, a=1.5, b=0.4, ns=(25, 50, 100, 200, 500, 1000, 1806)):
    n = np.asarray(ns, dtype=float)
    return n, power_law(n, ceiling, a, b)


# ------------------------------------------------------------- the fit -------
def test_fit_recovers_a_known_curve():
    n, y = synthetic_curve(ceiling=0.85, a=1.5, b=0.4)
    params, r2 = fit_curve(n, y)
    assert params is not None
    assert r2 > 0.999
    assert params[0] == pytest.approx(0.85, abs=0.02)
    assert params[2] == pytest.approx(0.4, abs=0.05)


def test_fit_is_robust_to_a_little_noise():
    n, y = synthetic_curve()
    y = y + np.random.default_rng(0).normal(0, 0.004, size=y.shape)
    params, r2 = fit_curve(n, y)
    assert params is not None
    assert r2 > 0.95
    assert params[0] == pytest.approx(0.85, abs=0.06)


def test_power_law_is_monotonically_increasing_and_saturating():
    n = np.array([10, 100, 1000, 10_000, 10**6, 10**12], dtype=float)
    y = power_law(n, 0.9, 1.5, 0.4)
    assert np.all(np.diff(y) > 0)
    assert np.all(y < 0.9)  # the ceiling is approached, never reached
    # b=0.4 means the shortfall decays as N**-0.4, which is slow: even a
    # million antibodies is still ~0.006 short. That slowness is the whole
    # reason the README's "tens of thousands" answers are so pessimistic.
    assert y[4] == pytest.approx(0.9, abs=1e-2)
    assert y[-1] == pytest.approx(0.9, abs=1e-4)


# ---------------------------------------------------- the ceiling cap --------
def test_ceiling_is_capped_a_fixed_margin_above_the_best_observed_score():
    """The guard that stops the fit parking the ceiling at 1.0.

    Without it, an unconstrained three-parameter fit on seven points predicts
    absurd sample sizes.  The cap has to actually bind on a curve that is still
    climbing steeply at the last measured point.
    """
    n = np.array([25, 50, 100, 200, 500], dtype=float)
    y = np.array([0.50, 0.55, 0.60, 0.65, 0.70])  # nowhere near saturating
    params, _ = fit_curve(n, y)
    assert params is not None
    assert params[0] <= y.max() + CEILING_HEADROOM + 1e-9


def test_upper_bound_argument_further_constrains_the_ceiling():
    n, y = synthetic_curve(ceiling=0.95, a=1.0, b=0.5)
    params, _ = fit_curve(n, y, upper=0.80)
    assert params is not None
    assert params[0] <= 0.80 + 1e-9


def test_fit_succeeds_even_when_the_cap_sits_below_the_data():
    """A degenerate but reachable configuration must not raise."""
    n, y = synthetic_curve(ceiling=0.9)
    params, r2 = fit_curve(n, y, upper=0.1)
    assert params is None or params[0] <= 0.1 + 1e-9


# ------------------------------------------------- refusing to over-claim ----
def test_fit_refuses_below_the_minimum_point_count():
    """Fewer than MIN_POINTS must yield no fit rather than a fragile one."""
    n = np.array([25, 50, 100], dtype=float)
    y = np.array([0.6, 0.65, 0.7])
    assert len(n) < MIN_POINTS
    params, r2 = fit_curve(n, y)
    assert params is None
    assert np.isnan(r2)


def test_non_finite_points_are_dropped_before_fitting():
    """Degenerate cells arrive as NaN and must not poison the fit."""
    n, y = synthetic_curve()
    y_with_holes = y.copy()
    y_with_holes[1] = np.nan
    params, r2 = fit_curve(n, y_with_holes)
    assert params is not None
    assert np.isfinite(r2)


def test_dropping_points_below_the_minimum_refuses_the_fit():
    n, y = synthetic_curve(ns=(25, 50, 100, 200, 500))
    y = y.astype(float)
    y[:2] = np.nan  # only three usable points left
    params, _ = fit_curve(n, y)
    assert params is None


def test_a_flat_curve_does_not_produce_a_confident_fit():
    n = np.array([25, 50, 100, 200, 500], dtype=float)
    y = np.full(5, 0.37)  # PPC-like: saturated and going nowhere
    params, r2 = fit_curve(n, y)
    if params is not None:
        # Whatever it fits, it must not claim a ceiling far above the data.
        assert params[0] <= 0.37 + CEILING_HEADROOM + 1e-9


# --------------------------------------------------------- the inversion -----
def test_n_for_target_inverts_the_power_law():
    params = (0.85, 1.5, 0.4)
    for target in (0.60, 0.70, 0.80):
        n = n_for_target(params, target)
        assert np.isfinite(n)
        assert power_law(n, *params) == pytest.approx(target, abs=1e-6)


def test_n_for_target_is_monotonic_in_the_target():
    params = (0.85, 1.5, 0.4)
    needs = [n_for_target(params, t) for t in (0.60, 0.70, 0.75, 0.80)]
    assert all(b > a for a, b in zip(needs, needs[1:]))


def test_a_target_at_or_above_the_ceiling_is_unreachable():
    """"Not by collecting more of this kind of data" has to be representable."""
    params = (0.85, 1.5, 0.4)
    assert n_for_target(params, 0.85) == float("inf")
    assert n_for_target(params, 0.99) == float("inf")


def test_round_trip_through_fit_and_inversion():
    n, y = synthetic_curve(ceiling=0.85, a=1.5, b=0.4)
    params, _ = fit_curve(n, y)
    target = 0.75
    needed = n_for_target(params, target)
    assert power_law(needed, *params) == pytest.approx(target, abs=1e-6)


# ------------------------------------------------------------ extrapolate ----
def _agg_frame(ceiling=0.85, a=1.5, b=0.4):
    n, y = synthetic_curve(ceiling, a, b)
    return pd.DataFrame(
        {
            "dataset": "SAbDab_Chen",
            "target": "developability",
            "split": "cluster",
            "features": "cheap",
            "n_train": n,
            "mean": y,
        }
    )


def test_extrapolate_reports_one_row_per_group_with_the_requested_targets():
    got = extrapolate(_agg_frame(), {"developability": [0.80, 0.85]})
    assert len(got) == 1
    row = got.iloc[0]
    assert {"n_for_0.8", "n_for_0.85"} <= set(got.columns)
    assert row["n_max_measured"] == 1806
    assert row["ceiling"] == pytest.approx(0.85, abs=0.02)
    assert row["fit_r2"] > 0.999
    assert row["n_points"] == 7


def test_extrapolate_flags_a_target_above_the_fitted_ceiling_as_nan():
    """A ceiling below the request means more data never gets you there."""
    got = extrapolate(_agg_frame(ceiling=0.75), {"developability": [0.95]})
    assert np.isnan(got.iloc[0]["n_for_0.95"])


def test_extrapolate_marks_a_long_extrapolation_unreliable():
    """The README leans on this: past ~10x the measured N, stop believing it."""
    frame = _agg_frame(ceiling=0.85, a=1.5, b=0.4)
    modest = extrapolate(frame, {"developability": [0.70]}).iloc[0]
    stretch = extrapolate(frame, {"developability": [0.8499]}).iloc[0]
    assert modest["reliable"]
    assert stretch["max_extrapolation_factor"] > modest["max_extrapolation_factor"]
    assert not stretch["reliable"]


def test_extrapolate_survives_a_group_it_cannot_fit():
    """Short groups must yield NaNs, not an exception or a dropped row."""
    frame = _agg_frame().head(3)
    got = extrapolate(frame, {"developability": [0.80]})
    assert len(got) == 1
    assert np.isnan(got.iloc[0]["ceiling"])
    assert np.isnan(got.iloc[0]["n_for_0.8"])


def test_extrapolate_separates_groups():
    a = _agg_frame()
    b = _agg_frame(ceiling=0.60)
    b["features"] = "random"
    got = extrapolate(pd.concat([a, b], ignore_index=True), {"developability": [0.55]})
    assert len(got) == 2
    ceilings = dict(zip(got["features"], got["ceiling"]))
    assert ceilings["cheap"] > ceilings["random"]


def test_extrapolate_requires_more_data_for_a_worse_curve():
    """Sanity: a lower-ceiling feature set needs more antibodies for a given score."""
    good = extrapolate(_agg_frame(ceiling=0.90), {"developability": [0.70]}).iloc[0]
    poor = extrapolate(_agg_frame(ceiling=0.75), {"developability": [0.70]}).iloc[0]
    assert poor["n_for_0.7"] > good["n_for_0.7"]
