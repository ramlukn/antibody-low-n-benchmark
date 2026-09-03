"""Paths, constants and the experiment grid.

Everything that another module might want to agree on lives here, so that a
change to (say) the list of training-set sizes propagates to the runner, the
plots and the notebook at once.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _p in (RAW, CACHE, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- datasets --
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM_DIM = 1280  # per chain; heavy+light concatenation is 2 * ESM_DIM

CHEN = "SAbDab_Chen"
TAP = "TAP"

TAP_TARGETS = ["CDR_Length", "PSH", "PPC", "PNC", "SFvCSP"]
TAP_TARGET_LABELS = {
    "CDR_Length": "Total CDR length",
    "PSH": "Patches of surface hydrophobicity",
    "PPC": "Patches of positive charge",
    "PNC": "Patches of negative charge",
    "SFvCSP": "Structural Fv charge symmetry",
}

# --------------------------------------------------------- experiment grid --
# Training-set sizes. TAP only has ~241 antibodies, so it gets a shorter grid.
N_GRID_CHEN = [25, 50, 100, 200, 500, 1000, "all"]
N_GRID_TAP = [25, 50, 100, 150, "all"]

N_SEEDS = 20
TEST_FRACTION = 0.25

FEATURE_SETS = ["random", "aac", "biophys", "cheap", "esm2", "esm2+cheap"]
HEADS = ["linear", "mlp", "gbm"]

# Sequence-identity threshold for grouping antibodies into clusters.
CLUSTER_IDENTITY = 0.80

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
