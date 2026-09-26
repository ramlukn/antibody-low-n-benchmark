"""Pooling invariants.

The pooling readout is where a per-residue model gets collapsed to one vector
per chain, and a masking mistake here would silently let padding or special
tokens into every embedding -- an error that shows up as a slightly worse score
rather than a crash, which is the worst kind.

These tests exercise the pooling arithmetic directly on synthetic hidden states,
so they need no model weights and no network.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="pooling maths needs torch")

from lown.embeddings import POOLINGS, _pool, pooled_dim  # noqa: E402


def hidden_and_mask(lengths, hidden_size=4, total_len=None):
    """Synthetic (B, L, D) hidden states plus a (B, L, 1) residue mask.

    Position 0 is BOS and position ``length + 1`` is EOS, mirroring how
    ``embed_sequences_multi`` builds its mask: both are excluded, as is padding.
    Real residues carry value ``+1``; every excluded position carries a large
    positive number, so any leak into a mean or a max is unmistakable.
    """
    b = len(lengths)
    total_len = total_len or max(lengths) + 2
    hidden = torch.full((b, total_len, hidden_size), 999.0)
    mask = torch.zeros((b, total_len, 1))
    for i, n in enumerate(lengths):
        hidden[i, 1 : n + 1] = 1.0
        mask[i, 1 : n + 1] = 1.0
    return hidden, mask


# ------------------------------------------------------------- dimensions ----
@pytest.mark.parametrize("pooling", POOLINGS)
def test_pooled_dim_matches_the_actual_output_width(pooling):
    hidden, mask = hidden_and_mask([5, 9])
    got = _pool(hidden, mask, pooling)
    assert got.shape == (2, pooled_dim(hidden.shape[-1], pooling))


def test_mean_plus_max_is_twice_as_wide():
    assert pooled_dim(1280, "mean+max") == 2560
    for p in ("mean", "max", "cls"):
        assert pooled_dim(1280, p) == 1280


def test_unknown_pooling_is_rejected():
    hidden, mask = hidden_and_mask([5])
    with pytest.raises(KeyError):
        _pool(hidden, mask, "attention")
    with pytest.raises(KeyError):
        pooled_dim(1280, "attention")


# ---------------------------------------------------------------- masking ----
@pytest.mark.parametrize("pooling", ["mean", "max", "mean+max"])
def test_special_tokens_and_padding_never_reach_the_pooled_vector(pooling):
    """The load-bearing masking test.

    Excluded positions hold 999 while real residues hold 1, so a leak is
    immediately visible in the pooled value.
    """
    hidden, mask = hidden_and_mask([3, 11], total_len=20)
    got = _pool(hidden, mask, pooling)
    assert torch.allclose(got, torch.ones_like(got)), f"{pooling} leaked a masked position"


def test_pooling_is_invariant_to_the_amount_of_padding():
    """Batching sorts by length, so the same chain meets different padding."""
    short, mask_short = hidden_and_mask([6], total_len=8)
    long, mask_long = hidden_and_mask([6], total_len=40)
    for pooling in ("mean", "max", "mean+max"):
        a = _pool(short, mask_short, pooling)
        b = _pool(long, mask_long, pooling)
        assert torch.allclose(a, b), f"{pooling} depends on padding width"


def test_pooling_is_independent_across_batch_members():
    """One long chain in a batch must not change a short chain's embedding."""
    alone, mask_alone = hidden_and_mask([5], total_len=30)
    together, mask_together = hidden_and_mask([5, 25], total_len=30)
    for pooling in ("mean", "max", "mean+max"):
        assert torch.allclose(
            _pool(alone, mask_alone, pooling), _pool(together, mask_together, pooling)[:1]
        ), f"{pooling} mixes batch members"


# -------------------------------------------------------------- semantics ----
def test_mean_is_the_average_over_real_residues_only():
    hidden = torch.zeros((1, 6, 2))
    mask = torch.zeros((1, 6, 1))
    hidden[0, 1] = 2.0
    hidden[0, 2] = 4.0
    mask[0, 1:3] = 1.0
    hidden[0, 0] = 100.0  # BOS, excluded
    hidden[0, 3] = 100.0  # EOS, excluded
    assert torch.allclose(_pool(hidden, mask, "mean"), torch.full((1, 2), 3.0))


def test_max_takes_the_largest_real_residue_per_dimension():
    hidden = torch.zeros((1, 5, 3))
    mask = torch.zeros((1, 5, 1))
    hidden[0, 1] = torch.tensor([1.0, 5.0, -2.0])
    hidden[0, 2] = torch.tensor([3.0, 0.0, -9.0])
    mask[0, 1:3] = 1.0
    assert torch.allclose(_pool(hidden, mask, "max"), torch.tensor([[3.0, 5.0, -2.0]]))


def test_max_handles_all_negative_activations():
    """A masked_fill with -inf must not turn a legitimately negative max into -inf."""
    hidden = torch.full((1, 4, 2), -7.0)
    mask = torch.zeros((1, 4, 1))
    mask[0, 1:3] = 1.0
    got = _pool(hidden, mask, "max")
    assert torch.isfinite(got).all()
    assert torch.allclose(got, torch.full((1, 2), -7.0))


def test_cls_reads_position_zero_regardless_of_the_mask():
    """BOS is excluded from mean/max but *is* the cls readout, by definition."""
    hidden, mask = hidden_and_mask([5])
    hidden[0, 0] = 42.0
    assert torch.allclose(_pool(hidden, mask, "cls"), torch.full((1, 4), 42.0))


def test_mean_plus_max_is_exactly_the_concatenation():
    hidden = torch.randn(3, 12, 5)
    mask = torch.zeros((3, 12, 1))
    mask[:, 1:9] = 1.0
    expected = torch.cat([_pool(hidden, mask, "mean"), _pool(hidden, mask, "max")], dim=-1)
    assert torch.allclose(_pool(hidden, mask, "mean+max"), expected)


def test_mean_and_max_differ_on_a_localised_signal():
    """The whole point of the ablation.

    One residue with a large activation -- a "patch" -- barely moves the mean but
    dominates the max. If these two readouts were interchangeable the ablation
    would be measuring nothing.
    """
    hidden = torch.zeros((1, 102, 1))
    mask = torch.zeros((1, 102, 1))
    mask[0, 1:101] = 1.0
    hidden[0, 50] = 100.0  # a single hot residue among a hundred
    mean = _pool(hidden, mask, "mean").item()
    maximum = _pool(hidden, mask, "max").item()
    assert mean == pytest.approx(1.0)
    assert maximum == pytest.approx(100.0)


# -------------------------------------------------------------- cache keys ----
def test_cache_paths_keep_mean_at_the_original_filename():
    """The committed 16 MB mean-pooled cache must stay valid."""
    from lown.embeddings import cache_paths

    npy, js = cache_paths("facebook/esm2_t33_650M_UR50D", "mean")
    assert npy.name == "esm2_t33_650M_UR50D.npy"
    assert js.name == "esm2_t33_650M_UR50D.seqs.json"


@pytest.mark.parametrize("pooling", ["max", "cls", "mean+max"])
def test_cache_paths_are_distinct_per_pooling(pooling):
    from lown.embeddings import cache_paths

    mean_npy, _ = cache_paths("facebook/esm2_t33_650M_UR50D", "mean")
    npy, _ = cache_paths("facebook/esm2_t33_650M_UR50D", pooling)
    assert npy != mean_npy
    assert "+" not in npy.name  # '+' is awkward in a filename
    assert npy.suffix == ".npy"


def test_cache_paths_are_distinct_across_models():
    from lown.embeddings import cache_paths

    a, _ = cache_paths("facebook/esm2_t33_650M_UR50D", "mean")
    b, _ = cache_paths("Exscientia/IgBert", "mean")
    assert a != b
