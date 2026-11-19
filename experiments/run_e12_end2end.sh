#!/usr/bin/env bash
# =============================================================================
#  E12 -- end-to-end teacher retrain, fanned across the idle GPUs.
#
#  Two grouped splits x four seeds = 8 runs, ~35 min each, one wave on 8 GPUs.
#  The seeds match the one-hot and autoencoder arms exactly, so the comparison
#  is paired: at a given seed every arm sees the same partition.
#
#  Usage:  bash experiments/run_e12_end2end.sh [n_parallel]
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
PY=/home/magesh/anaconda3/envs/molegen/bin/python
NPAR=${1:-8}
NGPU=$(nvidia-smi --list-gpus | wc -l)
LOG=experiments/results/logs
mkdir -p "$LOG"

JOBS=$(mktemp)
for split in cellpair pair; do
  for s in 42 1 2 3; do
    tag="e2e_${split}_s${s}"
    EXPORT=""
    # export learned embeddings once, from the reference run, so the existing
    # SHAP / symbolic / Flow 2 pipeline can consume them if the control clears
    if [[ "$split" == "cellpair" && "$s" == "42" ]]; then
      EXPORT="--export-embeddings clean_runs/2026-08-29_e2e_rerun/embeddings"
    fi
    echo "$PY experiments/e12_end2end_teacher.py --group-mode $split --seed $s --outdir experiments/results/$tag $EXPORT > $LOG/$tag.log 2>&1" >> "$JOBS"
  done
done

echo "queued $(wc -l < "$JOBS") runs, $NPAR at a time across $NGPU GPUs"
echo "logs -> $LOG"

nl -ba "$JOBS" | while read -r i cmd; do
  echo "CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) $cmd"
done | xargs -P "$NPAR" -I CMD bash -c CMD
rm -f "$JOBS"

echo
echo "all runs finished -- verdict:"
echo
$PY experiments/e12_aggregate.py
