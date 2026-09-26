"""Phase 1 -- run every unique antibody chain through frozen ESM-2 and cache.

No fine-tuning: a single forward pass, pooled over residues.  Chains are
deduplicated across both datasets first (many antibodies share a light
chain), then the cache is keyed by sequence so either dataset can look up
what it needs.

    python scripts/02_embed.py [--batch-size 8] [--device mps|cuda|cpu]
    python scripts/02_embed.py --poolings mean max cls mean+max

Extra poolings are free: the forward pass dominates the cost, so every readout
is computed from the same pass and cached separately.  See scripts/05_pooling.py
for what they are for.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lown.config import ESM_MODEL
from lown.data import all_sequences, load_chen, load_tap
from lown.embeddings import POOLINGS, cache_paths, embed_sequences_multi, save_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=ESM_MODEL)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--poolings", nargs="+", default=["mean"], choices=list(POOLINGS),
        help="readouts to cache; all are computed from one forward pass",
    )
    args = ap.parse_args()

    wanted = [
        p for p in args.poolings
        if args.force or not cache_paths(args.model, p)[0].exists()
    ]
    if not wanted:
        print(
            f"cache already present for {args.poolings}; pass --force to rebuild"
        )
        return
    skipped = sorted(set(args.poolings) - set(wanted))
    if skipped:
        print(f"already cached, skipping: {skipped}")

    chen, tap = load_chen(), load_tap()
    seqs = all_sequences([chen, tap])
    print(
        f"{len(chen)} SAbDab_Chen + {len(tap)} TAP antibodies "
        f"-> {len(seqs)} unique chains to embed"
    )

    print(f"pooling readouts to cache: {wanted}")
    t0 = time.time()
    mats = embed_sequences_multi(
        seqs, args.model, args.batch_size, args.device, poolings=tuple(wanted)
    )
    for pooling, mat in mats.items():
        save_cache(seqs, mat, args.model, pooling)
        npy, _ = cache_paths(args.model, pooling)
        print(f"  cached {pooling:9s} {mat.shape} -> {npy.name}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
