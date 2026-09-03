# Scripts

Run in order. Each one is independent and writes to `results/` or `figures/`.

| script | phase | what it does | runtime |
|---|---|---|---|
| `01_prepare_data.py` | 0-1 | download SAbDab_Chen and TAP, sanity-check the sequences, quantify near-duplicate leakage under both split types | seconds |
| `02_embed.py` | 1 | one frozen ESM-2 650M forward pass per unique chain, mean-pooled, cached to `data/cache/` | ~4 min (M-series Mac) |
| `03_baselines.py` | 2 | cross-validated scores and coefficients for the cheap descriptor blocks | ~1 min |
| `04_learning_curves.py` | 3-4 | the sweep: dataset x target x split x feature set x head x N x seed | ~40 min (12 cores) |
| `06_figures.py` | 5 | headline tables and every figure in the README | seconds |

`04` accepts `--quick` (3 seeds, linear head only) for a two-minute sanity check.
