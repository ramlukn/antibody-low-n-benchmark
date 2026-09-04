"""Phase 1 -- run every unique antibody chain through frozen ESM-2 and cache.

No fine-tuning: a single forward pass, mean-pooled over residues.  Chains are
deduplicated across both datasets first (many antibodies share a light
chain), then the cache is keyed by sequence so either dataset can look up
what it needs.

    python scripts/02_embed.py [--batch-size 8] [--device mps|cuda|cpu]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lown.config import ESM_MODEL
from lown.data import all_sequences, load_chen, load_tap
from lown.embeddings import cache_paths, embed_sequences, save_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=ESM_MODEL)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    npy, _ = cache_paths(args.model)
    if npy.exists() and not args.force:
        print(f"cache already present at {npy}; pass --force to rebuild")
        return

    chen, tap = load_chen(), load_tap()
    seqs = all_sequences([chen, tap])
    print(
        f"{len(chen)} SAbDab_Chen + {len(tap)} TAP antibodies "
        f"-> {len(seqs)} unique chains to embed"
    )

    t0 = time.time()
    mat = embed_sequences(seqs, args.model, args.batch_size, args.device)
    save_cache(seqs, mat, args.model)
    print(f"cached {mat.shape} to {npy} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
