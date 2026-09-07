#!/usr/bin/env bash
# Commit this project in four daily chunks. See COMMIT_PLAN.md.
#
#   ./scripts/commit_day.sh --list     show every commit, touch nothing
#   ./scripts/commit_day.sh 1          commit day 1's chunk
#   ./scripts/commit_day.sh --all      commit all four chunks now
set -euo pipefail

cd "$(dirname "$0")/.."

# --------------------------------------------------------------- the plan --
# Each entry: DAY|subject|body|space-separated paths
COMMITS=(
"1|chore: project scaffolding, pinned deps and Makefile|Package skeleton, the experiment grid in one place, and a Makefile so the whole benchmark is four commands.|.gitignore LICENSE requirements.txt Makefile src/lown/__init__.py src/lown/config.py scripts/README.md"

"1|feat(data): TDC loaders for SAbDab_Chen and TAP|TDC hands back a single Antibody column holding a stringified two-element container, and the two datasets disagree about which container (list vs numpy array). Parse the chains with a regex over amino-acid runs instead of trusting literal_eval. Heavy chain first, light second, verified across all 2,650 records.|src/lown/data.py"

"1|feat(embeddings): cache frozen ESM-2 650M chain embeddings|Forward pass only, no fine-tuning: mean-pool the last hidden state over real residues, excluding BOS/EOS and padding. Chains are deduplicated across both datasets first, taking 5,300 chain slots down to 3,238 unique sequences. Roughly four minutes on an M-series laptop.|src/lown/embeddings.py scripts/02_embed.py"

"2|feat(features): amino-acid composition and Biopython descriptors|Three cheap blocks a language model has to beat: length plus composition, global biophysics (net charge at pH 7, GRAVY, aromaticity, pI, instability), and motif-anchored CDR3 lengths. Deliberately includes the product of heavy- and light-chain net charge, which is close to the definition of one of the TAP targets; that is reported rather than hidden.|src/lown/features.py"

"2|feat(heads): linear, MLP and gradient-boosting heads with inner-CV tuning|Regularisation strength is chosen by cross-validation inside the training subsample only. At N=25 with a 20 percent positive rate a subsample can land with a single positive antibody, which leaves every stratified inner fold single-class; that case falls back to a fixed penalty rather than crashing.|src/lown/heads.py"

"2|exp(phase2): audit the cheap baselines and report their coefficients|Cross-validated scores per descriptor block under both split types, plus the strongest standardised coefficients. Fv hydrophobicity dominates developability; the heavy-times-light charge product dominates SFvCSP. Two of the five TAP metrics are close to unpredictable from sequence.|scripts/03_baselines.py results/baseline_scores.csv results/baseline_coefficients.csv"

"3|feat(splits): identity clustering and cluster-held-out partitions|Single-linkage clustering over a sequence-identity threshold, i.e. CD-HIT's idea without the dependency. Antibodies are linked if they are 90 percent identical over the whole Fv or 80 percent over their concatenated CDR3 loops; the union is conservative on purpose. Whole clusters are held out.|src/lown/splits.py"

"3|exp(phase0): dataset, clustering and leakage summaries|Quantifies the problem the cluster split exists to solve: 65 percent of a random SAbDab_Chen test set has an 80-percent-identity neighbour sitting in training. Also records that clustering the full Fv at 80 percent collapses half the dataset into one cluster, because antibody frameworks are conserved.|scripts/01_prepare_data.py results/dataset_summary.csv results/cdr_extraction_check.csv results/cluster_summary.csv results/leakage_summary.csv"

"3|feat(experiment): nested-subsample learning-curve sweep|For each seed the training subsamples are nested, so the N=25 set is a subset of the N=50 set and the curve is paired rather than independent at every point. The test set is fixed per seed and never subsampled. One job per feature-set/head/seed cell so the expensive cells spread across cores.|src/lown/experiment.py scripts/04_learning_curves.py"

"3|exp(phase3): full sweep results|Every fit in the benchmark as one tidy row: dataset, target, split, feature set, head, N, seed, metrics.|results/learning_curves.csv"

"4|feat(plots): learning-curve, split-gap and sample-efficiency figures|One colour per feature set held constant across every panel in the repo, so curves can be compared by eye between figures.|src/lown/plots.py src/lown/scaling.py scripts/06_figures.py figures results/learning_curves_aggregated.csv results/scaling_extrapolation.csv results/paired_esm_vs_cheap.csv results/headline.csv results/split_gap.csv results/crossover.csv results/sample_efficiency.csv"

"4|docs: README with learning curves and findings|Learning curves front and centre, what I found, and what I would do next with real wet-lab data.|README.md COMMIT_PLAN.md notebooks scripts/commit_day.sh"
)

usage() { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; }

list() {
  local day
  for entry in "${COMMITS[@]}"; do
    IFS='|' read -r day subject _ paths <<<"$entry"
    printf 'day %s  %s\n          %s\n' "$day" "$subject" "$paths"
  done
}

commit_day() {
  local want="$1" n=0 day subject body paths
  command -v git >/dev/null || { echo "git not found" >&2; exit 1; }
  git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository; run: git init" >&2; exit 1; }
  git config user.email >/dev/null 2>&1 || {
    echo "set your git identity first:" >&2
    echo "  git config user.name  'Your Name'" >&2
    echo "  git config user.email 'you@example.com'" >&2
    exit 1
  }

  for entry in "${COMMITS[@]}"; do
    IFS='|' read -r day subject body paths <<<"$entry"
    [ "$want" = "all" ] || [ "$day" = "$want" ] || continue

    local existing=()
    for p in $paths; do
      if [ -e "$p" ]; then existing+=("$p"); else echo "  skip (missing): $p" >&2; fi
    done
    [ ${#existing[@]} -gt 0 ] || { echo "  nothing to add for: $subject" >&2; continue; }

    git add -- "${existing[@]}"
    if git diff --cached --quiet; then
      echo "  already committed: $subject"
      continue
    fi

    # Wrap the body at 72 columns, the usual git convention.
    local wrapped
    wrapped=$(printf '%s\n' "$body" | fold -s -w 72 | sed 's/[[:space:]]*$//')
    local msg="$subject

$wrapped"

    git commit -q -m "$msg"
    n=$((n + 1))
    echo "  committed: $subject"
  done
  echo "day $want: $n commit(s)"
}

case "${1:-}" in
  --list|-l) list ;;
  --all|-a)  commit_day all ;;
  1|2|3|4)   commit_day "$1" ;;
  *)         usage; exit 1 ;;
esac
