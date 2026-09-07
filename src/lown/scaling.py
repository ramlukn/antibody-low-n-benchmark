"""Fit a power law to a learning curve and read off "how many more?".

The question a wet-lab planner actually asks is not "what is the AUC" but
"how many more antibodies do I have to measure before this is worth acting
on".  A saturating power law

    score(N) = ceiling - a * N ** (-b)

is the standard empirical form for learning curves and it inverts cleanly:

    N(target) = ((ceiling - target) / a) ** (-1 / b)

Two honest caveats travel with every number this produces.  Seven points and
three free parameters is a thin fit, and anything beyond the largest N
actually measured is extrapolation, not measurement.  Both are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def power_law(n, ceiling, a, b):
    return ceiling - a * np.power(n, -b)


# With five to seven points and three free parameters the ceiling is barely
# identifiable: an unconstrained fit happily parks it at 1.0 and then predicts
# absurd sample sizes.  Capping it a fixed margin above the best score actually
# observed keeps the extrapolation honest about what the data can support.
CEILING_HEADROOM = 0.30
MIN_POINTS = 5


def fit_curve(n: np.ndarray, y: np.ndarray, upper: float = 1.0):
    """Least-squares fit; returns (params, r2) or (None, nan) if it will not fit."""
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(n) & np.isfinite(y)
    n, y = n[ok], y[ok]
    if len(n) < MIN_POINTS:
        return None, float("nan")
    hi = float(min(upper, y.max() + CEILING_HEADROOM))
    lo = float(min(y.max(), hi - 1e-6))
    p0 = [min(float(y.max()) + 0.05, hi), 1.0, 0.3]
    try:
        params, _ = curve_fit(
            power_law, n, y, p0=p0, maxfev=20000,
            bounds=([lo, 1e-6, 1e-3], [hi, 1e4, 3.0]),
        )
    except Exception:
        return None, float("nan")
    resid = y - power_law(n, *params)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return params, r2


def n_for_target(params, target: float) -> float:
    ceiling, a, b = params
    if target >= ceiling:
        return float("inf")
    return float(((ceiling - target) / a) ** (-1.0 / b))


def extrapolate(agg_env: pd.DataFrame, targets: dict[str, list[float]]) -> pd.DataFrame:
    """One row per (dataset, target, split, feature set) with fitted ceiling and
    the N required to reach each requested score."""
    rows = []
    for (ds, tgt, sp, feat), g in agg_env.groupby(["dataset", "target", "split", "features"]):
        g = g.sort_values("n_train")
        upper = 1.0
        params, r2 = fit_curve(g.n_train.to_numpy(), g["mean"].to_numpy(), upper)
        row = dict(
            dataset=ds, target=tgt, split=sp, features=feat,
            n_max_measured=float(g.n_train.max()),
            score_at_n_max=float(g["mean"].iloc[-1]),
        )
        if params is None:
            row.update(ceiling=np.nan, fit_r2=np.nan, n_points=len(g))
            for t in targets.get(tgt, []):
                row[f"n_for_{t}"] = np.nan
        else:
            row.update(ceiling=float(params[0]), fit_r2=r2, n_points=len(g))
            worst = 0.0
            for t in targets.get(tgt, []):
                n = n_for_target(params, t)
                # NaN here means the fitted ceiling sits below the requested
                # score: more data of this kind never gets you there.
                row[f"n_for_{t}"] = n if np.isfinite(n) else np.nan
                if np.isfinite(n):
                    worst = max(worst, n / row["n_max_measured"])
            row["max_extrapolation_factor"] = round(worst, 1) if worst else np.nan
            row["reliable"] = bool(worst and worst <= 10.0)
        rows.append(row)
    return pd.DataFrame(rows)
