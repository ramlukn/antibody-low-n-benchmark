"""Feature invariants.

Two things matter here.  First, the reported dimensions (42 / 28 / 81) have to
match what the code actually produces, and the coefficient tables are only
readable if ``feature_names`` lines up with the matrix columns.  Second, the
regex CDR annotation is the acknowledged weak point of the benchmark, so its
behaviour -- including how it fails -- is pinned rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lown.features import (
    aac_features,
    biophys_features,
    build_matrix,
    cdr_features,
    cdr_h3,
    cdr_l3,
    cheap_features,
    feature_names,
    random_features,
)
from tests.conftest import make_heavy, make_light

BLOCKS = ["aac", "biophys", "cdr", "cheap"]

# The dimensions the README advertises.
ADVERTISED = {"aac": 42, "biophys": 28, "cheap": 81}


@pytest.mark.parametrize("block,width", ADVERTISED.items())
def test_block_widths_match_the_documented_dimensions(clonal, block, width):
    assert build_matrix(clonal, block).shape == (len(clonal), width)


@pytest.mark.parametrize("block", BLOCKS)
def test_feature_names_line_up_with_the_matrix_columns(clonal, block):
    """``baseline_coefficients.csv`` is unreadable if these ever drift apart."""
    matrix = build_matrix(clonal, block)
    assert len(feature_names(block)) == matrix.shape[1]


@pytest.mark.parametrize("block", BLOCKS)
def test_feature_names_are_unique(block):
    names = feature_names(block)
    assert len(names) == len(set(names))


def test_cheap_is_exactly_the_concatenation_of_its_blocks(clonal):
    h, l = clonal.loc[0, "heavy"], clonal.loc[0, "light"]
    expected = np.concatenate(
        [aac_features(h, l), biophys_features(h, l), cdr_features(h, l)]
    )
    assert cheap_features(h, l) == pytest.approx(expected)
    assert feature_names("cheap") == (
        feature_names("aac") + feature_names("biophys") + feature_names("cdr")
    )


@pytest.mark.parametrize("block", BLOCKS)
def test_features_are_finite(clonal, block):
    """A NaN here would silently poison a StandardScaler for a whole feature set."""
    assert np.isfinite(build_matrix(clonal, block)).all()


@pytest.mark.parametrize("block", BLOCKS)
def test_features_are_deterministic(clonal, block):
    assert np.array_equal(build_matrix(clonal, block), build_matrix(clonal, block))


def test_unknown_block_is_rejected(clonal):
    with pytest.raises(KeyError):
        build_matrix(clonal, "structure")
    with pytest.raises(KeyError):
        feature_names("structure")


# ------------------------------------------------- amino-acid composition ----
def test_aac_frequencies_sum_to_one_per_chain(clonal):
    matrix = build_matrix(clonal, "aac")
    # Layout is [len, 20 freqs] per chain, heavy then light.
    heavy_freqs, light_freqs = matrix[:, 1:21], matrix[:, 22:42]
    assert heavy_freqs.sum(axis=1) == pytest.approx(1.0, abs=1e-9)
    assert light_freqs.sum(axis=1) == pytest.approx(1.0, abs=1e-9)


def test_aac_encodes_length_in_hundreds_of_residues(clonal):
    matrix = build_matrix(clonal, "aac")
    expected = np.array([len(s) / 100.0 for s in clonal["heavy"]])
    assert matrix[:, 0] == pytest.approx(expected)


def test_aac_is_invariant_to_residue_order():
    """Composition is a bag of residues; anything positional must live elsewhere."""
    rng = np.random.default_rng(0)
    h = make_heavy("DEFGHIKLMNPQ", rng)
    l = make_light("YVTSRQPNM", rng)
    shuffled_h = "".join(rng.permutation(list(h)))
    a = aac_features(h, l)
    b = aac_features(shuffled_h, l)
    assert a[:21] == pytest.approx(b[:21])


# --------------------------------------------------- biophysical descriptors --
def test_charge_symmetry_term_is_the_product_of_the_chain_charges():
    """The feature that beats a language model on SFvCSP, asserted explicitly.

    The README is upfront that this term nearly encodes TAP's charge-symmetry
    target.  Keeping it tested means the claim stays checkable.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    rng = np.random.default_rng(0)
    h = make_heavy("DEFGHIKLMNPQ", rng)
    l = make_light("YVTSRQPNM", rng)

    names = feature_names("biophys")
    idx = names.index("Fv_qH_x_qL")
    qh = ProteinAnalysis(h).charge_at_pH(7.0)
    ql = ProteinAnalysis(l).charge_at_pH(7.0)
    assert biophys_features(h, l)[idx] == pytest.approx(qh * ql)


def test_charge_asymmetry_term_is_the_difference():
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    rng = np.random.default_rng(0)
    h, l = make_heavy("DEFGHIKLMNPQ", rng), make_light("YVTSRQPNM", rng)
    idx = feature_names("biophys").index("Fv_qH_minus_qL")
    qh = ProteinAnalysis(h).charge_at_pH(7.0)
    ql = ProteinAnalysis(l).charge_at_pH(7.0)
    assert biophys_features(h, l)[idx] == pytest.approx(qh - ql)


# ------------------------------------------------------- CDR3 annotation -----
def test_cdr_regexes_recover_a_planted_loop():
    rng = np.random.default_rng(0)
    for loop in ("DEFGHIKLMNPQ", "AKY", "DEFGHIKLMNPQRSTVYADEFG"):
        assert cdr_h3(make_heavy(loop, rng)) == loop
    for loop in ("YVTSRQPNM", "QQY"):
        assert cdr_l3(make_light(loop, rng)) == loop


def test_cdr_h3_capture_starts_one_residue_after_the_cysteine():
    """Pins an offset that makes the extracted loop *not* the canonical CDR-H3.

    ``_CDRH3`` allows one to three residues between the cysteine and the capture
    group, but the quantifier is lazy, so it always takes exactly one.  Planting
    the canonical ``C-A-R`` prefix therefore yields ``"R" + loop``, not ``loop``.

    This is a boundary convention, not a defect -- the README is explicit that
    these loops are "good enough to be a feature, not good enough to be a
    measurement" -- but it is precisely the kind of off-by-one that would be
    invisible in an aggregate score, so it is asserted rather than assumed.
    """
    rng = np.random.default_rng(0)
    loop = "DEFGHIKLMNPQ"
    canonical = make_heavy("R" + loop, rng)  # i.e. ...C-A-R-<loop>-WGQG...
    assert cdr_h3(canonical) == "R" + loop
    # Length is inflated by exactly the residues sitting inside the anchor.
    assert len(cdr_h3(canonical)) == len(loop) + 1


def test_cdr_extraction_prefers_the_rightmost_anchor():
    """Framework 3's cysteine is the last one, so the right-most match wins.

    A decoy cysteine upstream must not capture a spuriously long loop.
    """
    rng = np.random.default_rng(0)
    real = "DEFGHIKLMNPQ"
    seq = make_heavy(real, rng)
    decoy = seq[:40] + "CAA" + "DEFG" + "WGQG" + seq[40:]
    assert cdr_h3(decoy) == real


def test_cdr_extraction_returns_empty_string_when_the_motif_is_absent():
    """The documented 10% / 7% failure rate has to be a quiet "", not a crash."""
    assert cdr_h3("DEFGHIKLMNPQRSTVY" * 6) == ""
    assert cdr_l3("YVTSRQPNMLKIHGFED" * 6) == ""


def test_cdr_block_is_all_zeros_when_extraction_fails():
    """Missing loops must land as a clean zero row, not a NaN."""
    no_motif_h = "DEFGHIKLMNPQRSTVY" * 6
    no_motif_l = "YVTSRQPNMLKIHGFED" * 6
    got = cdr_features(no_motif_h, no_motif_l)
    assert np.isfinite(got).all()
    assert got == pytest.approx(np.zeros_like(got))


def test_cdr_total_length_is_the_sum_of_the_two_loops():
    rng = np.random.default_rng(0)
    h, l = make_heavy("DEFGHIKLMNPQ", rng), make_light("YVTSRQPNM", rng)
    got = cdr_features(h, l)
    idx = feature_names("cdr").index("CDR3_len_total")
    assert got[idx] == pytest.approx(12 + 9)


def test_cdr_length_feature_tracks_the_extracted_loop(clonal):
    matrix = build_matrix(clonal, "cdr")
    idx = feature_names("cdr").index("H3_cdr_len")
    expected = np.array([len(cdr_h3(s)) for s in clonal["heavy"]], dtype=float)
    assert matrix[:, idx] == pytest.approx(expected)


# ---------------------------------------------------------- random control ---
def test_random_features_are_a_fixed_draw_per_antibody(clonal):
    """The floor of every learning curve must be stable across the sweep."""
    a = random_features(clonal, dim=64, seed=0)
    b = random_features(clonal, dim=64, seed=0)
    assert a.shape == (len(clonal), 64)
    assert np.array_equal(a, b)


def test_random_features_carry_no_label_information(clonal):
    """Any apparent skill on these is optimism from the fitting procedure."""
    X = random_features(clonal, dim=64, seed=0)
    y = clonal["y"].to_numpy()
    correlations = [abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])]
    assert max(correlations) < 0.5


def test_random_features_differ_between_seeds(clonal):
    a = random_features(clonal, dim=16, seed=0)
    b = random_features(clonal, dim=16, seed=1)
    assert not np.array_equal(a, b)


# ----------------------------------------------------------- real data -------
@pytest.mark.real_data
def test_real_cdr_failure_rates_match_the_reported_table(real_chen):
    """Guards the numbers in ``results/cdr_extraction_check.csv``.

    Reported: 10.3% of heavy chains and 6.7% of light chains in SAbDab_Chen
    have no motif match.  A change to the regexes should have to update that
    table deliberately.
    """
    h_fail = np.mean([cdr_h3(s) == "" for s in real_chen["heavy"]])
    l_fail = np.mean([cdr_l3(s) == "" for s in real_chen["light"]])
    assert h_fail == pytest.approx(0.1029, abs=0.005)
    assert l_fail == pytest.approx(0.0668, abs=0.005)


@pytest.mark.real_data
def test_real_panel_features_are_finite(real_chen):
    subset = real_chen.head(200).reset_index(drop=True)
    for block in BLOCKS:
        assert np.isfinite(build_matrix(subset, block)).all(), block
