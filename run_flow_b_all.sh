#!/usr/bin/env bash
# Run Flow B (drug-level repurposing scored directly by the teacher) across
# every O'Neil cell line. Activate the molegen environment before running.
set -uo pipefail

CKPT="clean_runs/2026-04-29_full_rerun/synergy/oneil_synergy_runs/model.pt"
DATA="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl"
OUTROOT="${1:-clean_runs/flow_b}"
POOL="${2:-vocab}"

CELLS=("A2058" "A2780" "A375" "A427" "CAOV3" "DLD1" "ES2" "HCT116" "HT144"
       "HT29" "KPL1" "LOVO" "MDAMB436" "MSTO" "NCI-H460" "NCIH1650" "NCIH2122"
       "NCIH23" "NCIH520" "OV90" "OVCAR3" "PA1" "RKO" "RPMI7951" "SK-OV-3"
       "SKMEL30" "SKMES1" "SW-620" "SW837" "T-47D" "UACC62" "UWB1289" "VCAP"
       "ZR751")

mkdir -p "$OUTROOT"
echo "Flow B | pool=$POOL | ${#CELLS[@]} cell lines -> $OUTROOT"

ok=0; fail=0
for CL in "${CELLS[@]}"; do
  printf '  %-12s ' "$CL"
  if python flow_b_drug_repurposing.py \
       --checkpoint "$CKPT" \
       --dataset "$DATA" \
       --cell-line "$CL" \
       --outdir "$OUTROOT/$CL" \
       --candidate-pool "$POOL" \
       --min-z 1.0 \
       --min-delta 5.0 \
       --save-all-scores \
       > "$OUTROOT/${CL}.log" 2>&1; then
    n=$(grep -oP 'Hits: \K[0-9]+' "$OUTROOT/${CL}.log" | tail -1)
    echo "ok (${n:-0} hits)"
    ok=$((ok+1))
  else
    echo "FAILED (see $OUTROOT/${CL}.log)"
    fail=$((fail+1))
  fi
done

echo
echo "done: $ok ok, $fail failed"
[[ $fail -eq 0 ]] && python aggregate_flow_b.py --root "$OUTROOT"
