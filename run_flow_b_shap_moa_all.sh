#!/usr/bin/env bash
# Flow B (SHAP-driven, mechanism-matched) across every O'Neil cell line.
# Activate the molegen environment first.
set -uo pipefail

SHAP_ROOT="clean_runs/clean_runs/2026-07-10_full_rerun_inference/shap"
CKPT="clean_runs/2026-04-29_full_rerun/synergy/oneil_synergy_runs/model.pt"
DATA="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl"
MOA="${1:-drug_moa_cache_full.csv}"
OUTROOT="${2:-clean_runs/flow_b_shap_moa}"
TOPFRAG="${3:-50}"

mkdir -p "$OUTROOT"
echo "Flow B (SHAP + MoA) | moa=$MOA | top-fragments=$TOPFRAG -> $OUTROOT"
echo

ok=0; fail=0
for D in "$SHAP_ROOT"/*/; do
  CL="$(basename "$D")"
  FRAGS="$D/top_200_harmful_fragments.csv"
  [[ -f "$FRAGS" ]] || { echo "  $CL: no SHAP table, skipped"; continue; }

  printf '  %-12s ' "$CL"
  if python flow_b_shap_moa.py \
       --shap-harmful "$FRAGS" \
       --checkpoint "$CKPT" \
       --dataset "$DATA" \
       --cell-line "$CL" \
       --outdir "$OUTROOT/$CL" \
       --moa-cache "$MOA" \
       --top-fragments "$TOPFRAG" \
       > "$OUTROOT/${CL}.log" 2>&1; then
    line=$(grep -oP 'Hits: \K[0-9]+ \(MoA-high [0-9]+, in-vocab [0-9]+\)' "$OUTROOT/${CL}.log" | tail -1)
    echo "ok  ${line:-0}"
    ok=$((ok+1))
  else
    echo "FAILED -> $OUTROOT/${CL}.log"
    fail=$((fail+1))
  fi
done

echo
echo "done: $ok ok, $fail failed"
[[ $ok -gt 0 ]] && python aggregate_flow_b_shap_moa.py --root "$OUTROOT"
