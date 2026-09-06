"""Phase 0/1 -- download both datasets, sanity-check them, describe the splits.

Writes:
  results/dataset_summary.csv   per-dataset shape and label statistics
  results/cluster_summary.csv   clustering behaviour at several thresholds
  results/leakage_summary.csv   near-duplicate leakage, random vs cluster split

Run this first; everything downstream assumes the TDC cache exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lown.config import CHEN, RESULTS, TAP, TAP_TARGETS, TEST_FRACTION
from lown.data import load_chen, load_tap
from lown.features import cdr_h3, cdr_l3
from lown.splits import (
    DEFAULT_CRITERIA,
    cluster_report,
    identity_clusters,
    leakage_summary,
    make_splits,
)


def describe(chen: pd.DataFrame, tap: pd.DataFrame) -> pd.DataFrame:
    rows = [
        dict(
            dataset=CHEN,
            n=len(chen),
            task="classification",
            target="developability",
            summary=f"{int(chen.y.sum())} positive / {len(chen)} ({chen.y.mean():.1%})",
            unique_heavy=chen.heavy.nunique(),
            unique_light=chen.light.nunique(),
        )
    ]
    for t in TAP_TARGETS:
        rows.append(
            dict(
                dataset=TAP,
                n=len(tap),
                task="regression",
                target=t,
                summary=f"mean {tap[t].mean():.3g}, sd {tap[t].std():.3g}, "
                f"range [{tap[t].min():.3g}, {tap[t].max():.3g}]",
                unique_heavy=tap.heavy.nunique(),
                unique_light=tap.light.nunique(),
            )
        )
    return pd.DataFrame(rows)


def check_cdr(tap: pd.DataFrame, chen: pd.DataFrame) -> pd.DataFrame:
    """The regex CDR extractor has no ground truth here, so we validate it two
    ways: how often it fires at all, and how well the extracted CDR3 lengths
    track TAP's own total-CDR-length metric."""
    rows = []
    for name, df in [(TAP, tap), (CHEN, chen)]:
        h3 = df.heavy.map(cdr_h3)
        l3 = df.light.map(cdr_l3)
        rows.append(
            dict(
                dataset=name,
                h3_fail_frac=float((h3.str.len() == 0).mean()),
                l3_fail_frac=float((l3.str.len() == 0).mean()),
                h3_median_len=float(h3.str.len().median()),
                l3_median_len=float(l3.str.len().median()),
            )
        )
    out = pd.DataFrame(rows)
    tot = tap.heavy.map(lambda s: len(cdr_h3(s))) + tap.light.map(lambda s: len(cdr_l3(s)))
    out.loc[out.dataset == TAP, "spearman_vs_TAP_CDR_Length"] = spearmanr(
        tot, tap.CDR_Length
    ).statistic
    out.loc[out.dataset == TAP, "chainlen_spearman_vs_TAP_CDR_Length"] = spearmanr(
        tap.heavy.str.len() + tap.light.str.len(), tap.CDR_Length
    ).statistic
    return out


def cluster_table(chen: pd.DataFrame, tap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grids = [
        ("fv @ 80%", (("fv", 0.80),)),
        ("fv @ 90%", (("fv", 0.90),)),
        ("cdr3 @ 80%", (("cdr3", 0.80),)),
        ("union (default)", DEFAULT_CRITERIA),
    ]
    for name, df in [(CHEN, chen), (TAP, tap)]:
        for label, crit in grids:
            rows.append(dict(dataset=name, criterion=label, **cluster_report(identity_clusters(df, crit))))
    return pd.DataFrame(rows)


def leakage_table(chen: pd.DataFrame, tap: pd.DataFrame, n_seeds: int = 5) -> pd.DataFrame:
    rows = []
    for name, df in [(CHEN, chen), (TAP, tap)]:
        groups = identity_clusters(df)
        stratify = df.y.to_numpy() if "y" in df else None
        for mode in ("random", "cluster"):
            splits = make_splits(
                len(df), n_seeds, TEST_FRACTION, mode,
                groups=groups if mode == "cluster" else None,
                stratify=stratify if mode == "random" else None,
            )
            frac = [leakage_summary(df, tr, te) for tr, te in splits]
            rows.append(
                dict(
                    dataset=name,
                    split=mode,
                    mean_test_size=float(np.mean([len(te) for _, te in splits])),
                    test_with_near_duplicate_in_train=float(np.mean(frac)),
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    chen, tap = load_chen(), load_tap()

    tables = {
        "dataset_summary": describe(chen, tap),
        "cdr_extraction_check": check_cdr(tap, chen),
        "cluster_summary": cluster_table(chen, tap),
        "leakage_summary": leakage_table(chen, tap),
    }
    for name, tbl in tables.items():
        path = RESULTS / f"{name}.csv"
        tbl.to_csv(path, index=False)
        print(f"\n=== {name} -> {path.name}")
        print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
