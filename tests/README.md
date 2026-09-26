# Tests

```bash
make test        # or: pytest -q
```

201 tests, about eight seconds, no GPU and no downloads. The suite deliberately
does **not** depend on `torch`, `transformers` or `PyTDC` — see
`requirements-test.txt`. The point is that you can check the benchmark's
methodology is still intact without spending four minutes on a model forward
pass, so this runs in CI on every push.

The benchmark's value is entirely in whether its protocol is honest, so the
tests are organised around the claims the README makes rather than around the
modules that implement them.

| file | what it guards |
|---|---|
| `test_splits.py` | the leakage claim: cluster-held-out splits have **zero** near-duplicate leakage, random splits on a clonal panel have >80%, and clustering neither breaks a clonal family apart nor merges unrelated ones |
| `test_experiment.py` | the protocol claim: subsamples are genuinely **nested** (N=25 ⊂ N=50), the test set is byte-identical at every point on a curve, and every feature set and head sees the same rows within a seed — which is what licenses the paired Wilcoxon tests |
| `test_heads.py` | the no-leakage claim: no test-set statistic can reach the fit, plus honest degradation at N=25 with two positives |
| `test_features.py` | the advertised dimensions (42 / 28 / 81), `feature_names` lining up with the matrix columns, and the regex CDR annotation's behaviour **including how it fails** |
| `test_scaling.py` | the power-law fit recovers a known curve, the inversion is a true inverse, the ceiling cap actually binds, and the code refuses to fit rather than inventing a number |
| `test_embeddings.py` | the pooling arithmetic: no special token or padding may reach a pooled vector, and batching must not change a chain's embedding |
| `test_pooling_analysis.py` | the ablation's verdict helpers, on synthetic input where the right answer is known |
| `test_results.py` | every headline number in the README, read back off the committed tables in `results/` — including the pooling ablation |

## How the interesting tests work

Two techniques are worth knowing about before editing anything here.

**Observing the protocol instead of trusting it.** `test_experiment.py` builds a
feature matrix where row *i* is literally `[i]`, then substitutes a recording
stub for `fit_predict`. Whatever the protocol hands the model, the stub reports
the exact row indices. Nesting, the fixed test set and the shared subsample are
then checked directly rather than inferred from scores.

**Testing for leakage behaviourally.** `test_heads.py` never inspects the
pipeline to confirm the `StandardScaler` is fitted on training data only.
Instead it asserts that a given test row's prediction does not change when the
other test rows are removed, shuffled, duplicated, or joined by rows shifted by
1000 units. A scaler accidentally fitted on the combined data — the classic
low-N leak, and an easy one to reintroduce — fails all four immediately.

## Fixtures

`conftest.py` synthesises antibody panels rather than loading real ones, so the
suite is offline and deterministic.

- **`clonal`** — 12 families of 5, each sharing one CDR3 pair and differing by
  three framework point mutations. This is the structure that breaks a random
  split: an affinity-maturation panel built around a few leads, with the label
  assigned *per family* so a random split lets a model score well by memorising
  the family. Clustering recovers the 12 families exactly.
- **`distinct`** — 60 unrelated antibodies, all singletons. The TAP-like regime,
  where the leakage tax should be ~zero.
- **`real_chen`** — the actual SAbDab_Chen panel, or a skip when `data/raw` is
  empty. Used by the two tests marked `real_data`.

`make_heavy` / `make_light` plant a loop between the motif anchors the real
regexes look for. Note the single-residue spacer in `make_heavy`: `_CDRH3`
allows one to three residues after the framework-3 cysteine but the quantifier
is **lazy**, so it always takes exactly one. Planting the canonical `CAR` prefix
makes `cdr_h3` return `"R" + loop`, which is pinned by
`test_cdr_h3_capture_starts_one_residue_after_the_cysteine`.

## Two things the tests found

Neither changes a headline number, but both are the kind of thing that stays
invisible in an aggregate score:

1. **`cdr_h3` returns a loop offset by one residue** from the canonical CDR-H3,
   because the anchor quantifier is lazy. A boundary convention rather than a
   defect — the README is explicit that these loops are a feature and not a
   measurement — but it inflates every reported CDR-H3 length by one.
2. **Antibodies where both CDR3 regexes fail share an empty representation**,
   and `rapidfuzz.fuzz.ratio("", "") == 100`, so they all get linked into a
   single cluster. This affects 15 of 2,409 antibodies in SAbDab_Chen. The
   direction is the safe one — they are over-merged, so they land on the same
   side of the partition, which costs a little test-set size and can never
   manufacture leakage. `test_failed_cdr_extraction_merges_rather_than_leaks`
   fails if that direction ever flips.
