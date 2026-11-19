#!/usr/bin/env bash
# =============================================================================
#  E12d -- restarts on the two partitions that did NOT fail.
#
#  E12c retried only the failing partitions (2 and 3), which on its own is an
#  asymmetric protocol: retrying only your failures inflates the best-of-N.
#  This gives partitions 42 and 1 the same four initialisations, so the
#  best-of-four robustness check covers all four partitions equally.
#
#  Usage:  bash experiments/run_e12_restarts_healthy.sh [n_parallel]
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
PY=/home/magesh/anaconda3/envs/molegen/bin/python
NPAR=${1:-8}
NGPU=$(nvidia-smi --list-gpus | wc -l)
LOG=experiments/results/logs
mkdir -p "$LOG"

JOBS=$(mktemp)
for part in 42 1; do
  for init in 0 1 2 7; do
    tag="e2e_pair_p${part}_i${init}"
    echo "$PY experiments/e12_end2end_teacher.py --group-mode pair --seed $part --init-seed $init --outdir experiments/results/$tag > $LOG/$tag.log 2>&1" >> "$JOBS"
  done
done

echo "queued $(wc -l < "$JOBS") runs, $NPAR at a time across $NGPU GPUs"
nl -ba "$JOBS" | while read -r i cmd; do
  echo "CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) $cmd"
done | xargs -P "$NPAR" -I CMD bash -c CMD
rm -f "$JOBS"

echo
echo "=== best-of-four restarts, all four partitions ==="
$PY experiments/e12_restart_summary.py
