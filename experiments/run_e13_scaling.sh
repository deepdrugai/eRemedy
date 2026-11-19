#!/usr/bin/env bash
# =============================================================================
#  E13 -- vocabulary-scaling sweep on NCI-ALMANAC
#
#  Three drug representations x five vocabulary sizes x three seeds = 45 runs.
#
#  The design point: rows are capped at 8,500 for every vocabulary size, which
#  is what the smallest size can supply, so data volume is held constant and
#  only the number of distinct drugs varies. Drug subsets are nested within a
#  seed. Size 21 anchors the sweep at O'Neil's exact vocabulary size, so the
#  left-hand end of the curve should reproduce the O'Neil finding on a
#  completely different screen.
#
#  Usage:  bash experiments/run_e13_scaling.sh [n_parallel]
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
PY=/home/magesh/anaconda3/envs/molegen/bin/python
NPAR=${1:-8}
NGPU=$(nvidia-smi --list-gpus | wc -l)
LOG=experiments/results/logs
CACHE=clean_runs/almanac_cache
MAXROWS=8500
mkdir -p "$LOG"

[[ -f "$CACHE/meta.pkl" ]] || {
  echo "missing ALMANAC cache; run experiments/e13_build_almanac_cache.py first"
  exit 1; }

JOBS=$(mktemp)
for V in 21 30 40 50 60; do
  for arm in learned onehot ecfp; do
    for s in 42 1 2; do
      tag="scale_${arm}_v${V}_s${s}"
      echo "$PY experiments/e13_vocab_scaling.py --arm $arm --n-drugs $V --seed $s --max-rows $MAXROWS --cache $CACHE --outdir experiments/results/$tag > $LOG/$tag.log 2>&1" >> "$JOBS"
    done
  done
done

echo "queued $(wc -l < "$JOBS") runs, $NPAR at a time across $NGPU GPUs"
echo "rows held constant at $MAXROWS; vocabulary 21 -> 60"

nl -ba "$JOBS" | while read -r i cmd; do
  echo "CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) $cmd"
done | xargs -P "$NPAR" -I CMD bash -c CMD
rm -f "$JOBS"

echo
$PY experiments/e13_plot_scaling.py
