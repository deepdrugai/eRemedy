#!/usr/bin/env bash
# =============================================================================
#  E13b -- vocabulary scaling with PER-DRUG sampling held constant
#
#  The first sweep held total rows fixed at 8,500 across every vocabulary size.
#  Because ALMANAC measures close to all pairs, total rows grow roughly as V^2,
#  so fixing the total means each drug is seen ~3x less often at 60 drugs than
#  at 21 (about 810 appearances per drug falling to 283). That starves the
#  learned encoder of exactly the per-drug signal it needs, at the same time as
#  it widens the vocabulary -- which is the opposite of the intended contrast.
#
#  Here the row budget scales with the vocabulary instead: 400 rows per drug,
#  so 8,400 rows at V=21 rising to 24,000 at V=60. Appearances per drug stay
#  fixed near 800 while chemical diversity grows. This is the design in which a
#  threshold, if one exists, can actually appear.
#
#  Usage:  bash experiments/run_e13_scaling_perdrug.sh [n_parallel]
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
PY=/home/magesh/anaconda3/envs/molegen/bin/python
NPAR=${1:-8}
NGPU=$(nvidia-smi --list-gpus | wc -l)
LOG=experiments/results/logs
CACHE=clean_runs/almanac_cache
ROWS_PER_DRUG=400
mkdir -p "$LOG"

[[ -f "$CACHE/meta.pkl" ]] || { echo "missing ALMANAC cache"; exit 1; }

JOBS=$(mktemp)
for V in 21 30 40 50 60; do
  MAXROWS=$(( ROWS_PER_DRUG * V ))
  for arm in learned onehot ecfp; do
    for s in 42 1 2; do
      tag="perdrug_${arm}_v${V}_s${s}"
      echo "$PY experiments/e13_vocab_scaling.py --arm $arm --n-drugs $V --seed $s --max-rows $MAXROWS --cache $CACHE --outdir experiments/results/$tag > $LOG/$tag.log 2>&1" >> "$JOBS"
    done
  done
done

echo "queued $(wc -l < "$JOBS") runs, $NPAR at a time across $NGPU GPUs"
echo "rows = $ROWS_PER_DRUG x vocabulary (8,400 -> 24,000); per-drug sampling fixed"

nl -ba "$JOBS" | while read -r i cmd; do
  echo "CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) $cmd"
done | xargs -P "$NPAR" -I CMD bash -c CMD
rm -f "$JOBS"

echo
$PY experiments/e13_plot_scaling.py \
  --prefix perdrug \
  --subtitle "rows scaled with vocabulary (400 per drug)" \
  --out vocabulary_threshold_perdrug
