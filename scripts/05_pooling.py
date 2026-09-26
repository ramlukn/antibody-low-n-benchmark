"""Phase 4b -- is mean pooling the reason frozen ESM-2 loses?

The headline result of this benchmark is that frozen ESM-2 embeddings do not
beat Biopython descriptors in the low-N regime.  The first objection to that is
not about the model, it is about the readout: mean pooling over ~120 residues is
the cheapest possible summary of a per-residue representation, and a hydrophobic
patch is a *local* property.  Maybe the representation is fine and the average
is what destroys it.

That objection makes a specific, falsifiable prediction.  Three of TAP's five
targets are explicitly patch properties -- patches of surface hydrophobicity
(PSH), of positive charge (PPC) and of negative charge (PNC).  Max pooling asks
"does a residue like this exist anywhere in the chain", which is much closer to
what those metrics measure.  So if pooling is the problem, **max pooling should
beat mean pooling on exactly those three targets**, and the effect should be
weaker or absent on the two that are not patch properties (SFvCSP, CDR_Length)
and on SAbDab developability.

Four readouts of the same frozen model, all from one forward pass:

  mean      the original: average over real residues
  max       per-dimension maximum over real residues -- the patch detector
  cls       the BOS token, frozen.  A control: it has never been trained to
            summarise anything, so it should be the worst of the four
  mean+max  both, concatenated (5120-d per antibody)

The cheap descriptors ride along as the reference line, so every number here is
directly comparable to the main experiment.  Cluster-held-out split only: the
random split's leakage would flatter every variant equally and tell us nothing
about which readout generalises.

    python scripts/05_pooling.py
    python scripts/05_pooling.py --quick            # 3 seeds, linear only
    python scripts/05_pooling.py --heads linear     # skip the MLP
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from lown.config import CHEN, N_SEEDS, RESULTS, TAP, TAP_TARGETS
from lown.data import load_chen, load_tap
from lown.embeddings import antibody_matrix, cache_paths, load_cache
from lown.experiment import chen_task, cluster_groups, run_task, tap_tasks
from lown.features import build_matrix

# Which readouts to compare, and the label each one carries in the output.
VARIANTS = {
    "mean": "esm2_mean",
    "max": "esm2_max",
    "cls": "esm2_cls",
    "mean+max": "esm2_mean+max",
}
REFERENCE = "cheap"

# The three TAP targets that are patch properties, i.e. where the "mean pooling
# throws away where things are" objection predicts max pooling should win.
PATCH_TARGETS = ("PSH", "PPC", "PNC")


def build_bank(df, lookups: dict[str, dict]) -> dict[str, np.ndarray]:
    """Cheap descriptors plus one matrix per pooling readout."""
    bank = {REFERENCE: build_matrix(df, "cheap")}
    for pooling, label in VARIANTS.items():
        bank[label] = antibody_matrix(df, lookups[pooling])
    return bank


def load_lookups(poolings) -> dict[str, dict]:
    missing = [p for p in poolings if not cache_paths(pooling=p)[0].exists()]
    if missing:
        raise SystemExit(
            "No embedding cache for pooling(s) "
            + ", ".join(missing)
            + ".\nRun: python scripts/02_embed.py --poolings "
            + " ".join(VARIANTS)
        )
    return {p: load_cache(pooling=p) for p in poolings}


def unify(df: pd.DataFrame) -> pd.DataFrame:
    """One comparable score column across classification and regression."""
    df = df.copy()
    for col in ("auc", "spearman"):
        if col not in df:
            df[col] = np.nan
    is_clf = df.task == "classification"
    df["value"] = np.where(is_clf, df["auc"], df["spearman"])
    return df


def paired_vs(raw: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Seed-paired delta of every variant against ``baseline``, within each head.

    Identical machinery to ``06_figures.paired_tests``: every seed uses the same
    split and the same nested subsample for every feature set, so a Wilcoxon
    signed-rank test over the seeds is the right thing to run.
    """
    from scipy.stats import wilcoxon

    rows = []
    for (ds, tgt, sp, hd, nr), g in raw.groupby(
        ["dataset", "target", "split", "head", "n_request"]
    ):
        piv = g.pivot_table(index="seed", columns="features", values="value")
        if baseline not in piv:
            continue
        for label in piv.columns:
            if label == baseline:
                continue
            d = (piv[label] - piv[baseline]).dropna()
            if len(d) < 5:
                continue
            try:
                p = float(wilcoxon(d).pvalue)
            except ValueError:  # all differences identical / zero
                p = np.nan
            rows.append(
                dict(dataset=ds, target=tgt, split=sp, head=hd, n_request=nr,
                     comparison=f"{label} - {baseline}", features=label,
                     baseline=baseline, n_seeds=int(len(d)),
                     mean_delta=float(d.mean()), sd_delta=float(d.std()),
                     p_wilcoxon=p)
            )
    return pd.DataFrame(rows)


def patch_verdict(paired: pd.DataFrame) -> pd.DataFrame:
    """Does max pooling help more on patch targets than on the rest?

    The whole point of the ablation, reduced to one table: the mean seed-paired
    advantage of max over mean pooling at the largest N, split by whether the
    target is a patch property.
    """
    at_max_n = paired[
        (paired.features == "esm2_max")
        & (paired.baseline == "esm2_mean")
        & (paired.n_request == "all")
    ].copy()
    at_max_n["target_kind"] = np.where(
        at_max_n.target.isin(PATCH_TARGETS), "patch", "not patch"
    )
    return (
        at_max_n.groupby(["target_kind", "head"])
        .agg(
            n_targets=("target", "nunique"),
            mean_delta=("mean_delta", "mean"),
            worst=("mean_delta", "min"),
            best=("mean_delta", "max"),
            n_significant_wins=("p_wilcoxon", lambda s: int(
                ((s < 0.05) & (at_max_n.loc[s.index, "mean_delta"] > 0)).sum()
            )),
        )
        .reset_index()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--quick", action="store_true", help="3 seeds, linear head only")
    ap.add_argument("--heads", nargs="*", default=["linear", "mlp"])
    ap.add_argument("--datasets", nargs="*", default=[CHEN, TAP])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = 3 if args.quick else args.seeds
    heads = ["linear"] if args.quick else args.heads
    out_path = Path(args.out) if args.out else RESULTS / (
        "pooling_ablation_quick.csv" if args.quick else "pooling_ablation.csv"
    )

    lookups = load_lookups(list(VARIANTS))
    frames = []
    t0 = time.time()

    if CHEN in args.datasets:
        chen = load_chen()
        bank = build_bank(chen, lookups)
        print(f"\n### {CHEN} / cluster split  "
              f"({', '.join(f'{k}:{v.shape[1]}d' for k, v in bank.items())})")
        frames.append(
            run_task(chen_task(), chen, bank, "cluster", groups=cluster_groups(chen),
                     n_seeds=seeds, heads=heads, n_jobs=args.n_jobs)
        )

    if TAP in args.datasets:
        tap = load_tap()
        bank = build_bank(tap, lookups)
        groups = cluster_groups(tap)
        for task in tap_tasks():
            print(f"\n### TAP / {task.target} / cluster split")
            frames.append(
                run_task(task, tap, bank, "cluster", groups=groups,
                         n_seeds=seeds, heads=heads, n_jobs=args.n_jobs)
            )

    res = pd.concat(frames, ignore_index=True)
    res.to_csv(out_path, index=False)
    print(f"\n{len(res)} fits -> {out_path} in {time.time() - t0:.0f}s")

    raw = unify(res)

    # Against mean pooling: does any other readout beat the original?
    vs_mean = paired_vs(raw, "esm2_mean")
    # Against the cheap descriptors: does any readout change the headline?
    vs_cheap = paired_vs(raw, REFERENCE)
    paired = pd.concat([vs_mean, vs_cheap], ignore_index=True)

    if args.quick:
        print(paired.to_string(index=False))
        return

    paired.to_csv(RESULTS / "pooling_paired.csv", index=False)
    verdict = patch_verdict(paired)
    verdict.to_csv(RESULTS / "pooling_patch_verdict.csv", index=False)

    # One panel per target, best head per point -- the same optimistic envelope
    # the main figures use, applied identically to every readout.
    from lown.plots import aggregate, best_head_envelope, pooling_figure

    curves = []
    for metric in ("auc", "spearman"):
        part = raw[raw[metric].notna()]
        if len(part):
            curves.append(aggregate(part, metric))
    agg = pd.concat(curves, ignore_index=True)
    pooling_figure(best_head_envelope(agg), TAP_TARGETS)

    print("\n=== every readout minus mean pooling, at the largest N "
          "(seed-paired, Wilcoxon)")
    show = vs_mean[vs_mean.n_request == "all"].sort_values(
        ["dataset", "target", "head", "features"]
    )
    print(show[["dataset", "target", "head", "features", "mean_delta",
                "p_wilcoxon"]].to_string(index=False))

    print("\n=== the prediction under test: max pooling should help on patch "
          "targets specifically")
    print(verdict.to_string(index=False))

    print("\n=== best readout minus cheap descriptors, at the largest N")
    best = (
        vs_cheap[vs_cheap.n_request == "all"]
        .sort_values("mean_delta", ascending=False)
        .groupby(["dataset", "target", "head"])
        .head(1)
        .sort_values(["dataset", "target", "head"])
    )
    print(best[["dataset", "target", "head", "features", "mean_delta",
                "p_wilcoxon"]].to_string(index=False))
    print(f"\ntables -> {RESULTS / 'pooling_paired.csv'}, "
          f"{RESULTS / 'pooling_patch_verdict.csv'}")


if __name__ == "__main__":
    main()
