"""Phase 3 + 4 -- the whole benchmark: learning curves under both split types.

Sweeps dataset x target x split mode x feature set x head x training-set
size x seed and writes one tidy row per fit to results/learning_curves.csv.

    python scripts/04_learning_curves.py                 # full run
    python scripts/04_learning_curves.py --quick         # 3 seeds, sanity check
    python scripts/04_learning_curves.py --datasets TAP  # one dataset only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from lown.config import CHEN, HEADS, N_SEEDS, RESULTS, TAP
from lown.data import load_chen, load_tap
from lown.embeddings import load_cache
from lown.experiment import chen_task, cluster_groups, feature_bank, run_task, tap_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--quick", action="store_true", help="3 seeds, linear head only")
    ap.add_argument("--datasets", nargs="*", default=[CHEN, TAP])
    ap.add_argument("--splits", nargs="*", default=["random", "cluster"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = 3 if args.quick else args.seeds
    heads = ["linear"] if args.quick else HEADS
    out_path = Path(args.out) if args.out else RESULTS / (
        "learning_curves_quick.csv" if args.quick else "learning_curves.csv"
    )

    lookup = load_cache()
    frames = []
    t0 = time.time()

    if CHEN in args.datasets:
        chen = load_chen()
        bank = feature_bank(chen, lookup)
        groups = cluster_groups(chen)
        for split in args.splits:
            print(f"\n### {CHEN} / {split} split")
            frames.append(
                run_task(chen_task(), chen, bank, split,
                         groups=groups if split == "cluster" else None,
                         n_seeds=seeds, heads=heads, n_jobs=args.n_jobs)
            )

    if TAP in args.datasets:
        tap = load_tap()
        bank = feature_bank(tap, lookup)
        groups = cluster_groups(tap)
        for task in tap_tasks():
            for split in args.splits:
                print(f"\n### TAP / {task.target} / {split} split")
                frames.append(
                    run_task(task, tap, bank, split,
                             groups=groups if split == "cluster" else None,
                             n_seeds=seeds, heads=heads, n_jobs=args.n_jobs)
                )

    res = pd.concat(frames, ignore_index=True)
    res.to_csv(out_path, index=False)
    print(f"\n{len(res)} fits -> {out_path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
