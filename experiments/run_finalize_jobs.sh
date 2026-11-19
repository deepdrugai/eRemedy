#!/usr/bin/env bash
#
# Launch the two remaining job sets for the paper, fanned across the idle GPUs.
#
#   set 1  E8 seed replication   3 extra seeds x {none,drug} x {cellpair,pair}
#          Existing seed-42 runs stay in place, giving 4 seeds in total.
#
#   set 2  E10 DeepSynergy one-hot drug control
#          Tests whether the identity confound is architecture-independent.
#
# Usage:  bash experiments/run_finalize_jobs.sh [n_parallel]
#
set -u

cd "$(dirname "$0")/.."
PY=/home/magesh/anaconda3/envs/molegen/bin/python
NPAR=${1:-8}
NGPU=$(nvidia-smi --list-gpus | wc -l)
LOG=experiments/results/logs
mkdir -p "$LOG"

JOBS=$(mktemp)

for s in 1 2 3; do
  for split in cellpair pair; do
    for ab in none drug; do
      tag="onehot_${ab}_${split}_s${s}"
      echo "$PY experiments/e8_onehot_controls.py --ablate $ab --group-mode $split --seed $s --outdir experiments/results/$tag > $LOG/$tag.log 2>&1" >> "$JOBS"
    done
  done
done

for split in cellpair pair; do
  tag="deepsynergy_onehot_${split}"
  echo "$PY experiments/e9_deepsynergy_baseline.py --split $split --drug-features onehot --outdir experiments/results/$tag > $LOG/$tag.log 2>&1" >> "$JOBS"
done

echo "queued $(wc -l < "$JOBS") runs, $NPAR at a time across $NGPU GPUs"
echo "logs -> $LOG"

nl -ba "$JOBS" | while read -r i cmd; do
  echo "CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) $cmd"
done | xargs -P "$NPAR" -I CMD bash -c CMD

echo "all runs finished"
rm -f "$JOBS"
