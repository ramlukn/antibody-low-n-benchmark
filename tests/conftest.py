"""Shared fixtures.

Everything here is synthetic and offline.  The test suite must run on a clean
checkout with no TDC download, no ESM-2 weights and no network, because the
invariants it guards are properties of the *code*, not of the data.  The few
tests that genuinely want the real panels are marked ``real_data`` and skip
themselves when ``data/raw`` is empty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Frameworks are deliberately distinct between families here.  Real antibody
# frameworks are near-identical, which is exactly why the production criteria
# union whole-Fv identity with CDR3 identity -- but a unit test wants families
# that the clustering can actually tell apart, so that "did it separate them"
# has a definite right answer.
_FRAMEWORK_ALPHABET = "ADEFGHIKLMNPQRSTVY"  # no C/W, so we control motif placement


def _framework(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list(_FRAMEWORK_ALPHABET), size=length))


def make_heavy(cdr3: str, rng: np.random.Generator, fw_seed: int = 0) -> str:
    """A heavy chain whose CDR-H3 is exactly ``cdr3`` under ``features.cdr_h3``.

    The regex anchors on the framework-3 cysteine (plus one to three residues)
    through the J-segment ``W-G-x-G`` motif, and the right-most match wins.

    Note the single-residue spacer.  ``[A-Z]{1,3}?`` is lazy, so it consumes the
    minimum of one residue and the capture group starts immediately after it --
    planting the canonical ``CAR`` prefix would make ``cdr_h3`` return
    ``"R" + cdr3``.  One spacer residue is what makes the round trip exact.
    """
    fw_rng = np.random.default_rng(fw_seed)
    return (
        _framework(fw_rng, 95)      # frameworks 1-3
        + "C" + "A" + cdr3 + "WGQG"  # the motif-anchored CDR-H3
        + _framework(fw_rng, 11)    # framework 4
    )


def make_light(cdr3: str, rng: np.random.Generator, fw_seed: int = 0) -> str:
    """A light chain whose CDR-L3 is exactly ``cdr3`` under ``features.cdr_l3``."""
    fw_rng = np.random.default_rng(fw_seed + 500)
    return (
        _framework(fw_rng, 88)
        + "C" + cdr3 + "FGQG"
        + _framework(fw_rng, 10)
    )


def _mutate(seq: str, n_mut: int, rng: np.random.Generator) -> str:
    """Point-mutate ``n_mut`` framework positions, leaving the motifs intact."""
    chars = list(seq)
    # Only touch the first 80 residues: well clear of the CDR3 motifs, so a
    # clonal variant keeps the same CDR3 and stays >=90% identical overall.
    positions = rng.choice(80, size=n_mut, replace=False)
    for p in positions:
        chars[p] = str(rng.choice(list(_FRAMEWORK_ALPHABET)))
    return "".join(chars)


def clonal_panel(
    n_families: int = 12,
    per_family: int = 5,
    seed: int = 0,
    positive_rate: float = 0.25,
) -> pd.DataFrame:
    """A panel of near-duplicate families, the structure that breaks random splits.

    Every family shares one CDR3 pair and one framework, differing only by a
    handful of framework point mutations -- i.e. an affinity-maturation panel
    built around a few leads.  Members of a family are >=90% identical over
    the Fv and 100% identical over their CDR3s, so any honest split has to keep
    a whole family on one side of the partition.

    The label is assigned *per family*, which is the pessimistic case: a random
    split then lets a model score well by memorising the family rather than
    learning anything.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_families):
        h3 = "".join(rng.choice(list(_FRAMEWORK_ALPHABET), size=12))
        l3 = "".join(rng.choice(list(_FRAMEWORK_ALPHABET), size=9))
        heavy0 = make_heavy(h3, rng, fw_seed=f)
        light0 = make_light(l3, rng, fw_seed=f)
        label = int(rng.random() < positive_rate)
        for m in range(per_family):
            rows.append(
                dict(
                    id=f"fam{f}_m{m}",
                    family=f,
                    heavy=heavy0 if m == 0 else _mutate(heavy0, 3, rng),
                    light=light0 if m == 0 else _mutate(light0, 3, rng),
                    y=label,
                )
            )
    return pd.DataFrame(rows)


def distinct_panel(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A panel with no near-duplicates: every antibody its own family.

    This is the TAP-like case, where the leakage tax should be roughly zero.
    """
    return clonal_panel(n_families=n, per_family=1, seed=seed)


@pytest.fixture
def clonal():
    return clonal_panel()


@pytest.fixture
def distinct():
    return distinct_panel()


@pytest.fixture
def real_chen():
    """The actual SAbDab_Chen panel, or a skip when it has not been downloaded."""
    from lown.config import RAW

    if not (RAW / "sabdab_chen.tab").exists():
        pytest.skip("SAbDab_Chen not downloaded; run `make data`")
    from lown.data import load_chen

    return load_chen()
