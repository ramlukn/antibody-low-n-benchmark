"""Frozen ESM-2 embeddings for antibody chains.

No fine-tuning, no training: one forward pass per unique chain, mean-pooled
over real residues (special tokens and padding excluded), cached to disk.
Heavy and light vectors are concatenated at lookup time to give one
2 x 1280 = 2560-dimensional vector per antibody.

Chains are deduplicated across both datasets before embedding -- many
antibodies share a light chain, so this is a real saving.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CACHE, ESM_MODEL


def _device(prefer: str | None = None) -> str:
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_sequences(
    seqs: list[str],
    model_name: str = ESM_MODEL,
    batch_size: int = 8,
    device: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Mean-pooled last-hidden-state embedding for each sequence."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = _device(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()

    # Sort by length so batches are homogeneous; undo the permutation at the end.
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = [seqs[i] for i in idx]
            enc = tok(batch, return_tensors="pt", padding=True).to(dev)
            hidden = model(**enc).last_hidden_state  # (B, L, D)

            # Drop BOS/EOS as well as padding: attention_mask keeps both.
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0
            lengths = enc["attention_mask"].sum(1) - 1
            mask[torch.arange(mask.size(0), device=dev), lengths] = 0
            mask = mask.unsqueeze(-1).to(hidden.dtype)

            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out[idx] = pooled.float().cpu().numpy()

            if verbose and (start // batch_size) % 25 == 0:
                print(f"  {min(start + batch_size, len(order))}/{len(order)}", flush=True)

    return out


def cache_paths(model_name: str = ESM_MODEL) -> tuple[Path, Path]:
    tag = model_name.split("/")[-1]
    return CACHE / f"{tag}.npy", CACHE / f"{tag}.seqs.json"


def save_cache(seqs: list[str], mat: np.ndarray, model_name: str = ESM_MODEL) -> None:
    npy, js = cache_paths(model_name)
    np.save(npy, mat)
    js.write_text(json.dumps(seqs))


def load_cache(model_name: str = ESM_MODEL) -> dict[str, np.ndarray]:
    npy, js = cache_paths(model_name)
    if not npy.exists():
        raise FileNotFoundError(
            f"No embedding cache at {npy}. Run `python scripts/02_embed.py` first."
        )
    mat = np.load(npy)
    seqs = json.loads(js.read_text())
    return {s: mat[i] for i, s in enumerate(seqs)}


def antibody_matrix(df, lookup: dict[str, np.ndarray]) -> np.ndarray:
    """Heavy || light for each row of a dataset frame."""
    return np.vstack(
        [np.concatenate([lookup[h], lookup[l]]) for h, l in zip(df["heavy"], df["light"])]
    )
