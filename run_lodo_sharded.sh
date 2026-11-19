#!/usr/bin/env bash
# =============================================================================
#  run_lodo_sharded.sh
#
#  Leave-One-Drug-Out validation of the synergy teacher, sharded across the
#  free GPUs. Each shard trains a fresh model per held-out drug.
#
#  Runtime: ~1 h per drug at 200 epochs; 21 drugs over 5 GPUs ~= 4-5 hours.
#
#  Usage (run inside tmux, with `conda activate molegen` already done):
#    ./run_lodo_sharded.sh
#    ./run_lodo_sharded.sh 50            # quicker pass at 50 epochs
#
#  Monitor:   tail -f clean_runs/lodo/shard_*.log
#  Merge:     python merge_lodo.py
# =============================================================================

set -uo pipefail

EPOCHS="${1:-200}"
OUTROOT="clean_runs/lodo"
mkdir -p "$OUTROOT"

# GPUs with no resident memory at time of writing
GPUS=(0 4 5 6 7)

# 21 drugs, interleaved so each shard gets a mix of large and small row counts.
# Declared as separate arrays because "mitomycin C" contains a space.
SHARD0=("MK-1775" "391210-10-9" "gemcitabine" "cyclophosphamide" "vinorelbine")
SHARD1=("915019-65-7" "MK-4541" "vinblastine" "etoposide")
SHARD2=("Dasatinib" "temozolomide" "7-Ethyl-10-hydroxycamptothecin" "metformin")
SHARD3=("MRK003" "geldanamycin" "mitomycin C" "methotrexate")
SHARD4=("Veliparib" "MK-2206" "Dinaciclib" "Bortezomib")

echo "Launching LODO: 5 shards, ${EPOCHS} epochs each"
echo "Output root: $OUTROOT"
echo

PIDS=()
for i in 0 1 2 3 4; do
  GPU="${GPUS[$i]}"
  LOG="$OUTROOT/shard_${i}.log"
  # Expand the i-th shard array by name
  declare -n SHARD="SHARD$i"
  echo "  shard $i -> GPU $GPU : ${SHARD[*]}"
  CUDA_VISIBLE_DEVICES="$GPU" nohup python validate_lodo.py \
      --drugs "${SHARD[@]}" \
      --epochs "$EPOCHS" \
      --outdir "$OUTROOT/shard_$i" \
      > "$LOG" 2>&1 &
  PIDS+=($!)
  unset -n SHARD
done

echo
echo "PIDs: ${PIDS[*]}"
echo "Waiting for all shards to finish ..."
FAIL=0
for p in "${PIDS[@]}"; do
  wait "$p" || FAIL=$((FAIL+1))
done

echo
if [[ $FAIL -eq 0 ]]; then
  echo "All shards completed successfully."
else
  echo "WARNING: $FAIL shard(s) exited non-zero -- check $OUTROOT/shard_*.log"
fi
echo "Now run:  python merge_lodo.py"
