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


# --------------------------------------------------------------- pooling ----
# Mean pooling is the cheapest possible readout of a per-residue model, and it
# is the obvious objection to this benchmark's headline result: perhaps ESM-2
# loses at low N because averaging over 120 residues destroys the signal, not
# because the representation is uninformative.
#
# Three of TAP's five targets are explicitly *patch* properties -- patches of
# surface hydrophobicity, positive charge, negative charge -- i.e. local and
# spatial.  Max pooling over residues is the natural readout for "does a patch
# like this exist anywhere", so if pooling were the problem, max should beat
# mean on exactly those three targets.  That is a directional prediction, which
# makes it worth testing rather than conceding.
POOLINGS = ("mean", "max", "cls", "mean+max")


def pooled_dim(hidden_size: int, pooling: str) -> int:
    if pooling == "mean+max":
        return 2 * hidden_size
    if pooling in ("mean", "max", "cls"):
        return hidden_size
    raise KeyError(pooling)


def _pool(hidden, mask, pooling: str):
    """Pool (B, L, D) hidden states down to (B, D'), honouring ``mask``.

    ``mask`` is (B, L, 1) and already excludes BOS, EOS and padding, so no
    special token ever contributes to a mean or a max.
    """
    import torch

    if pooling == "cls":
        # The BOS representation, which is what a fine-tuned classifier would
        # normally read.  Frozen, it has never been trained to summarise
        # anything, so this is included as a control rather than a contender.
        return hidden[:, 0]
    if pooling == "mean":
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    if pooling == "max":
        # Masked positions must not win the max, so push them to -inf.
        return hidden.masked_fill(mask == 0, float("-inf")).max(dim=1).values
    if pooling == "mean+max":
        return torch.cat([_pool(hidden, mask, "mean"), _pool(hidden, mask, "max")], dim=-1)
    raise KeyError(pooling)


def embed_sequences_multi(
    seqs: list[str],
    model_name: str = ESM_MODEL,
    batch_size: int = 8,
    device: str | None = None,
    poolings: tuple[str, ...] = ("mean",),
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Every requested pooling, from a *single* forward pass per sequence.

    The forward pass dominates the cost, so computing four readouts together
    costs what one costs.  Returns ``{pooling: matrix}``.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    for p in poolings:
        pooled_dim(1, p)  # fail fast on a typo, before loading 650M parameters

    dev = _device(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    hidden_size = model.config.hidden_size

    # Sort by length so batches are homogeneous; undo the permutation at the end.
    order = np.argsort([len(s) for s in seqs])
    out = {
        p: np.zeros((len(seqs), pooled_dim(hidden_size, p)), dtype=np.float32)
        for p in poolings
    }

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

            for p in poolings:
                out[p][idx] = _pool(hidden, mask, p).float().cpu().numpy()

            if verbose and (start // batch_size) % 25 == 0:
                print(f"  {min(start + batch_size, len(order))}/{len(order)}", flush=True)

    return out


def embed_sequences(
    seqs: list[str],
    model_name: str = ESM_MODEL,
    batch_size: int = 8,
    device: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Mean-pooled last-hidden-state embedding for each sequence."""
    return embed_sequences_multi(
        seqs, model_name, batch_size, device, poolings=("mean",), verbose=verbose
    )["mean"]


def cache_paths(model_name: str = ESM_MODEL, pooling: str = "mean") -> tuple[Path, Path]:
    """Cache location for one (model, pooling) pair.

    Mean pooling keeps the original unsuffixed filename, so the committed
    16 MB cache from the main experiment stays valid.
    """
    tag = model_name.split("/")[-1]
    if pooling != "mean":
        tag = f"{tag}.{pooling.replace('+', '_')}"
    return CACHE / f"{tag}.npy", CACHE / f"{tag}.seqs.json"


def save_cache(
    seqs: list[str], mat: np.ndarray, model_name: str = ESM_MODEL, pooling: str = "mean"
) -> None:
    npy, js = cache_paths(model_name, pooling)
    np.save(npy, mat)
    js.write_text(json.dumps(seqs))


def load_cache(model_name: str = ESM_MODEL, pooling: str = "mean") -> dict[str, np.ndarray]:
    npy, js = cache_paths(model_name, pooling)
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
