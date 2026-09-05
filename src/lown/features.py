"""Cheap, non-learned featurisations of an antibody Fv.

These are the comparators that a protein language model has to beat.  Three
blocks, all computable in seconds on a laptop:

``aac``      sequence length + amino-acid composition, per chain.
``biophys``  Biopython global descriptors (net charge at pH 7, GRAVY,
             aromaticity, pI, instability, secondary-structure fractions),
             per chain, plus a handful of whole-Fv terms.
``cdr``      regex-anchored CDR-H3 / CDR-L3 lengths and their composition.

``cheap`` is the concatenation of all three: the "domain knowledge, no GPU"
baseline.  One of its features (the product of heavy- and light-chain net
charge) is by construction close to the definition of the TAP SFvCSP target.
That is deliberate and is reported as such -- when the property you are
predicting is a known physical quantity, a descriptor that encodes it will
beat a learned representation, and pretending otherwise makes the benchmark
dishonest.
"""
from __future__ import annotations

import re

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from .config import AMINO_ACIDS

# --------------------------------------------------------------------------
# CDR extraction
#
# Proper CDR annotation wants ANARCI (HMM alignment to a germline numbering
# scheme).  We deliberately avoid that dependency and use the classical
# motif-anchored regexes instead: the CDR3 loops sit between the conserved
# cysteine that closes framework 3 and the J-segment motif that opens
# framework 4 (W-G-x-G for heavy, F-G-x-G for light).  ``validate_cdr`` in
# scripts/01_prepare_data.py checks the result against TAP's CDR_Length.
# --------------------------------------------------------------------------
_CDRH3 = re.compile(r"C[A-Z]{1,3}?(?P<cdr>[A-Z]{1,32}?)W[GA][QKRSTG]G")
_CDRL3 = re.compile(r"C(?P<cdr>[A-Z]{1,20}?)F[GA][QGSA]G")


def _extract(seq: str, pattern: re.Pattern) -> str:
    """Right-most motif match wins: framework 3's cysteine is the last one."""
    best = ""
    for m in pattern.finditer(seq):
        best = m.group("cdr")
    return best


def cdr_h3(seq: str) -> str:
    return _extract(seq, _CDRH3)


def cdr_l3(seq: str) -> str:
    return _extract(seq, _CDRL3)


# ------------------------------------------------------------------ blocks --
def aac_block(seq: str) -> np.ndarray:
    """Length (in units of 100 residues) + 20 amino-acid frequencies."""
    n = max(len(seq), 1)
    counts = np.array([seq.count(a) for a in AMINO_ACIDS], dtype=float)
    return np.concatenate([[n / 100.0], counts / n])


AAC_NAMES = ["len"] + [f"frac_{a}" for a in AMINO_ACIDS]


def biophys_block(seq: str) -> np.ndarray:
    pa = ProteinAnalysis(seq)
    helix, turn, sheet = pa.secondary_structure_fraction()
    return np.array(
        [
            pa.molecular_weight() / 1000.0,
            pa.aromaticity(),
            pa.instability_index(),
            pa.gravy(),
            pa.isoelectric_point(),
            pa.charge_at_pH(7.0),
            pa.charge_at_pH(5.5),
            helix,
            turn,
            sheet,
            pa.molar_extinction_coefficient()[1] / 1000.0,
        ],
        dtype=float,
    )


BIOPHYS_NAMES = [
    "mw_kda",
    "aromaticity",
    "instability",
    "gravy",
    "pI",
    "charge_pH7",
    "charge_pH5.5",
    "frac_helix",
    "frac_turn",
    "frac_sheet",
    "ext_coeff",
]


def cdr_block(seq: str, pattern: re.Pattern) -> np.ndarray:
    cdr = _extract(seq, pattern)
    if not cdr:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    pa = ProteinAnalysis(cdr)
    return np.array(
        [
            len(cdr),
            pa.gravy(),
            pa.charge_at_pH(7.0),
            pa.aromaticity(),
            sum(cdr.count(a) for a in "DE") - sum(cdr.count(a) for a in "KR"),
        ],
        dtype=float,
    )


CDR_NAMES = ["cdr_len", "cdr_gravy", "cdr_charge", "cdr_aromaticity", "cdr_net_acidic"]


# --------------------------------------------------------------- assembly ---
def _pair(fn, heavy: str, light: str) -> np.ndarray:
    return np.concatenate([fn(heavy), fn(light)])


def aac_features(heavy: str, light: str) -> np.ndarray:
    return _pair(aac_block, heavy, light)


def biophys_features(heavy: str, light: str) -> np.ndarray:
    per_chain = _pair(biophys_block, heavy, light)
    fv = ProteinAnalysis(heavy + light)
    qh = ProteinAnalysis(heavy).charge_at_pH(7.0)
    ql = ProteinAnalysis(light).charge_at_pH(7.0)
    whole = np.array(
        [
            fv.gravy(),
            fv.charge_at_pH(7.0),
            fv.isoelectric_point(),
            qh * ql,          # charge-symmetry style term
            qh - ql,          # charge asymmetry
            len(heavy) / max(len(light), 1),
        ]
    )
    return np.concatenate([per_chain, whole])


def cdr_features(heavy: str, light: str) -> np.ndarray:
    h = cdr_block(heavy, _CDRH3)
    l = cdr_block(light, _CDRL3)
    return np.concatenate([h, l, [h[0] + l[0]]])


def cheap_features(heavy: str, light: str) -> np.ndarray:
    return np.concatenate(
        [aac_features(heavy, light), biophys_features(heavy, light), cdr_features(heavy, light)]
    )


def feature_names(block: str) -> list[str]:
    h = [f"H_{n}" for n in AAC_NAMES]
    l = [f"L_{n}" for n in AAC_NAMES]
    if block == "aac":
        return h + l
    if block == "biophys":
        return (
            [f"H_{n}" for n in BIOPHYS_NAMES]
            + [f"L_{n}" for n in BIOPHYS_NAMES]
            + ["Fv_gravy", "Fv_charge_pH7", "Fv_pI", "Fv_qH_x_qL", "Fv_qH_minus_qL", "Fv_len_ratio"]
        )
    if block == "cdr":
        return (
            [f"H3_{n}" for n in CDR_NAMES] + [f"L3_{n}" for n in CDR_NAMES] + ["CDR3_len_total"]
        )
    if block == "cheap":
        return feature_names("aac") + feature_names("biophys") + feature_names("cdr")
    raise KeyError(block)


BUILDERS = {
    "aac": aac_features,
    "biophys": biophys_features,
    "cdr": cdr_features,
    "cheap": cheap_features,
}


def build_matrix(df, block: str) -> np.ndarray:
    fn = BUILDERS[block]
    return np.vstack([fn(h, l) for h, l in zip(df["heavy"], df["light"])])


def random_features(df, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Information-free control: Gaussian noise, one fixed draw per antibody.

    Any apparent skill on these features is pure optimism from the fitting
    procedure, so this line is the honest floor of every learning curve.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(df), dim))
