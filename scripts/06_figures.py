"""Phase 5 -- turn results/learning_curves.csv into figures and headline tables.

Writes figures/fig1..fig5 and four tables under results/:
  headline.csv           best score per target under each split
  crossover.csv          where frozen ESM-2 overtakes the cheap descriptors
  sample_efficiency.csv  N needed to reach a useful score
  split_gap.csv          random-split minus cluster-split, the honesty tax
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from lown.config import RESULTS, TAP_TARGETS
from lown.scaling import extrapolate
from lown.plots import (
    aggregate,
    best_head_envelope,
    chen_figure,
    crossover_figure,
    gap_figure,
    sample_efficiency_figure,
    scaling_figure,
    tap_figure,
)

# A score worth having: clearly better than chance and stable enough to act on.
USEFUL = {"classification": 0.70, "regression": 0.50}

# Scores to extrapolate a required N for, per target.
SCALING_TARGETS = {
    "developability": [0.80, 0.85, 0.90],
    "PSH": [0.6, 0.7],
    "PNC": [0.6, 0.7],
    "PPC": [0.6, 0.7],
    "SFvCSP": [0.8, 0.9],
    "CDR_Length": [0.99],
}


def load(path: Path | None = None) -> pd.DataFrame:
    path = path or RESULTS / "learning_curves.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/04_learning_curves.py first")
    return pd.read_csv(path)


def unify(df: pd.DataFrame) -> pd.DataFrame:
    """One comparable score column across the classification and regression tasks."""
    df = df.copy()
    for col in ("auc", "spearman"):
        if col not in df:
            df[col] = np.nan
    is_clf = df.task == "classification"
    df["metric"] = np.where(is_clf, "ROC-AUC", "Spearman rho")
    df["value"] = np.where(is_clf, df["auc"], df["spearman"])
    return df


def headline(agg_env: pd.DataFrame) -> pd.DataFrame:
    """Best feature set x head at the largest N, per target and split."""
    full = agg_env[agg_env.n_request == "all"]
    idx = full.groupby(["dataset", "target", "split"])["mean"].idxmax()
    out = full.loc[idx, ["dataset", "target", "split", "features", "head", "n_train", "mean", "std"]]
    return out.sort_values(["dataset", "target", "split"]).reset_index(drop=True)


def split_gap(head_tbl: pd.DataFrame) -> pd.DataFrame:
    piv = head_tbl.pivot_table(index=["dataset", "target"], columns="split", values="mean")
    piv = piv.rename(columns={"random": "random_"}).reset_index()
    piv["gap"] = piv["random_"] - piv["cluster"]
    return piv.sort_values("gap", ascending=False).reset_index(drop=True)


def crossover(agg_env: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per split: ESM-2 minus cheap at the largest N, and the smallest N at
    which ESM-2's mean first exceeds cheap's mean."""
    rows, deltas = [], []
    for (ds, tgt, sp), g in agg_env.groupby(["dataset", "target", "split"]):
        esm = g[g.features == "esm2"].set_index("n_request")["mean"]
        cheap = g[g.features == "cheap"].set_index("n_request")["mean"]
        common = [n for n in _order(g) if n in esm.index and n in cheap.index]
        wins = [n for n in common if esm[n] > cheap[n]]
        first = None
        for n in common:
            if all(esm[m] > cheap[m] for m in common[common.index(n):]):
                first = n
                break
        rows.append(
            dict(dataset=ds, target=tgt, split=sp,
                 esm_beats_cheap_from_N=first if first is not None else "never",
                 n_points_esm_ahead=len(wins), n_points=len(common))
        )
        last = common[-1]
        deltas.append(dict(dataset=ds, target=tgt, split=sp,
                           delta=float(esm[last] - cheap[last])))
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def paired_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Seed-paired ESM-2 minus cheap, within each head.

    Every seed uses the same split and the same nested subsample for both
    feature sets, so the comparison is paired and a Wilcoxon signed-rank test
    over the 20 seeds is the right thing to run.  Comparing best-of-heads
    envelopes instead would smuggle in a selection effect.
    """
    from scipy.stats import wilcoxon

    rows = []
    for (ds, tgt, sp, hd, nr), g in raw.groupby(
        ["dataset", "target", "split", "head", "n_request"]
    ):
        piv = g.pivot_table(index="seed", columns="features", values="value")
        if "esm2" not in piv or "cheap" not in piv:
            continue
        d = (piv["esm2"] - piv["cheap"]).dropna()
        if len(d) < 5:
            continue
        try:
            p = float(wilcoxon(d).pvalue)
        except ValueError:  # all differences identical / zero
            p = np.nan
        rows.append(
            dict(dataset=ds, target=tgt, split=sp, head=hd, n_request=nr,
                 n_seeds=int(len(d)), mean_delta=float(d.mean()),
                 sd_delta=float(d.std()), p_wilcoxon=p)
        )
    return pd.DataFrame(rows)


def _order(g: pd.DataFrame) -> list[str]:
    o = g.groupby("n_request")["n_train"].mean().sort_values()
    return list(o.index)


def sample_efficiency(agg_env: pd.DataFrame, tasks: dict) -> pd.DataFrame:
    rows = []
    for (ds, tgt, sp, feat), g in agg_env.groupby(["dataset", "target", "split", "features"]):
        thr = USEFUL[tasks[(ds, tgt)]]
        g = g.assign(_x=g.n_train).sort_values("_x")
        hit = g[g["mean"] >= thr]
        rows.append(
            dict(dataset=ds, target=tgt, split=sp, features=feat,
                 threshold=thr,
                 n_required=float(hit.n_train.iloc[0]) if len(hit) else np.nan)
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="alternative learning_curves.csv")
    args = ap.parse_args()
    raw = unify(load(Path(args.input) if args.input else None))
    tasks = {(r.dataset, r.target): r.task for r in raw[["dataset", "target", "task"]].drop_duplicates().itertuples()}

    agg = aggregate(raw, "value")
    agg_env = best_head_envelope(agg)

    head_tbl = headline(agg_env)
    gap = split_gap(head_tbl)
    cross, delta = crossover(agg_env)
    eff = sample_efficiency(agg_env, tasks)
    scal = extrapolate(agg_env, SCALING_TARGETS)
    paired = paired_tests(raw)

    agg.to_csv(RESULTS / "learning_curves_aggregated.csv", index=False)
    head_tbl.to_csv(RESULTS / "headline.csv", index=False)
    gap.to_csv(RESULTS / "split_gap.csv", index=False)
    cross.to_csv(RESULTS / "crossover.csv", index=False)
    eff.to_csv(RESULTS / "sample_efficiency.csv", index=False)
    scal.to_csv(RESULTS / "scaling_extrapolation.csv", index=False)
    paired.to_csv(RESULTS / "paired_esm_vs_cheap.csv", index=False)

    chen_figure(agg)
    tap_figure(agg, TAP_TARGETS)
    gap_figure(gap)
    crossover_figure(delta.groupby(["target", "split"]).delta.mean().reset_index())
    sample_efficiency_figure(eff[eff.split == "cluster"].sort_values("target"))
    scaling_figure(agg_env, "SAbDab_Chen", ["developability"])
    scaling_figure(agg_env, "TAP", ["PSH", "SFvCSP"],
                   path=RESULTS.parent / "figures" / "fig7_scaling_tap.png")

    print("\n=== headline: best score at the largest N")
    print(head_tbl.round(3).to_string(index=False))
    print("\n=== the honesty tax: random split minus cluster-held-out split")
    print(gap.round(3).to_string(index=False))
    print("\n=== does frozen ESM-2 overtake the cheap descriptors?")
    print(cross.to_string(index=False))
    print("\n=== ESM-2 minus cheap, seed-paired, at the largest N (Wilcoxon over 20 seeds)")
    print(
        paired[paired.n_request == "all"]
        .sort_values(["dataset", "target", "split"])
        [["target", "split", "head", "mean_delta", "sd_delta", "p_wilcoxon"]]
        .round(4).to_string(index=False)
    )
    print("\n=== power-law extrapolation, cluster-held-out split (EXTRAPOLATION, not measurement)")
    cols = [c for c in scal.columns if c.startswith("n_for_") or c in
            ("target", "features", "n_max_measured", "score_at_n_max", "ceiling", "fit_r2")]
    print(
        scal[(scal.split == "cluster") & (scal.features.isin(["cheap", "esm2"]))][cols]
        .round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
