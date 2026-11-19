#!/usr/bin/env bash
# =============================================================================
#  run_pipeline_all_celllines.sh
#
#  Runs Steps 5–10 (SHAP → Symbolic → Annotate → Filter → Docker →
#  Rescore → Repurposing) sequentially for every ONEIL cell line
#  EXCEPT A2058 and HCT116 (already completed).
#
#  Usage:
#    chmod +x run_pipeline_all_celllines.sh
#    nohup ./run_pipeline_all_celllines.sh > pipeline_overnight.log 2>&1 &
#
#  Progress is written to pipeline_overnight.log (one section per cell line).
#  If a cell line fails the script logs the error, skips to the next one,
#  and writes a FAILED marker in the output directory so you can re-run
#  individual lines later:
#    ./run_pipeline_all_celllines.sh MCF7
# =============================================================================

set -uo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
RUN_ROOT="/home/magesh/ecomb2/clean_runs/2026-04-29_full_rerun"
RUN_ROOT_OUTPUT_CLEAN="/home/magesh/ecomb2/clean_runs/clean_runs/2026-07-10_full_rerun_inference"

DATASET="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl"
MOL_EMBS="$RUN_ROOT/molecule_ae/molecule_smiles_embeddings.pkl"
FRAG_EMBS="$RUN_ROOT/fragment_ae/fragment_smiles_embeddings.pkl"
FRAG_MAP="$RUN_ROOT/fragment_ae/smiles_to_fragments.pkl"
DRUG_MAP="/home/magesh/ecomb2/everything_drug_smiles_map.csv"
CHECKPOINT="$RUN_ROOT/synergy/oneil_synergy_runs/model.pt"
MOL_ENC="$RUN_ROOT/molecule_ae/drug_smiles_encoder.pt"
FRAG_ENC="$RUN_ROOT/fragment_ae/fragment_smiles_encoder.pt"
SMILES_DIR="/home/magesh/ecomb2/dataset_EVERYTHING_SMILES"
FRAG_DIR="/home/magesh/ecomb2/dataset-smi-frag"
MOA_CACHE="/home/magesh/ecomb2/drug_moa_cache.csv"


# ── Cell lines to process (all ONEIL lines except A2058 and HCT116) ──────────
ALL_CELL_LINES=(
  A375 A427 CAOV3 DLD1 ES2 HT144 HT29 KPL1 LOVO
  MDAMB436 MSTO NCI-H460 NCIH1650 NCIH2122 NCIH23 NCIH520
  OV90 OVCAR3 PA1 RKO RPMI7951 SK-OV-3 SKMEL30 SKMES1
  SW-620 SW837 T-47D UACC62 UWB1289 VCAP ZR751
)
# ALL_CELL_LINES=(
#   A2780
# )

# Allow overriding from CLI: ./run_pipeline_all_celllines.sh MCF7 A375
if [[ $# -gt 0 ]]; then
  ALL_CELL_LINES=("$@")
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*"; }
fail() { log "ERROR: $*"; }

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL=${#ALL_CELL_LINES[@]}
PASS=0; FAIL_LIST=()

for CELL_LINE in "${ALL_CELL_LINES[@]}"; do
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "START  $CELL_LINE  ($(( PASS + ${#FAIL_LIST[@]} + 1 )) / $TOTAL)"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  SHAP_OUT="$RUN_ROOT_OUTPUT_CLEAN/shap/$CELL_LINE"
  SYM_OUT="$RUN_ROOT_OUTPUT_CLEAN/symbolic/$CELL_LINE"
  RESCORE_OUT="$RUN_ROOT_OUTPUT_CLEAN/rescored/$CELL_LINE"
  REPURPOSE_OUT="$RUN_ROOT_OUTPUT_CLEAN/repurposing/$CELL_LINE"

  RANKED_CSV="$SYM_OUT/replacement_candidates_ranked.csv"
  WITH_SDF_CSV="$SYM_OUT/replacement_candidates_ranked_with_fragment_sdf.csv"
  TOP_CSV="$SYM_OUT/replacement_candidates_top200_positive_with_fragment_sdf.csv"
  COMPACT_CSV="$SYM_OUT/replacement_candidates_top200_positive_compact.csv"
  SCORED_CSV="$SYM_OUT/replacement_candidates_top200_positive_with_fragment_sdf_scored.csv"

  # Track whether this cell line succeeded all steps
  CL_OK=1

  run_step() {
    local STEP_NAME="$1"; shift
    if [[ $CL_OK -eq 0 ]]; then return; fi   # skip if a previous step failed
    log "  → Step $STEP_NAME"
    if ! "$@"; then
      fail "Step $STEP_NAME failed for $CELL_LINE — skipping remaining steps."
      CL_OK=0
    fi
  }

  # ── Step 5: SHAP ─────────────────────────────────────────────────────────
  run_step "5-SHAP" \
    python synergy_shap_analysis.py \
      --checkpoint "$CHECKPOINT" \
      --dataset "$DATASET" \
      --molecule-embeddings "$MOL_EMBS" \
      --fragment-embeddings "$FRAG_EMBS" \
      --fragment-map "$FRAG_MAP" \
      --drug-smiles-map "$DRUG_MAP" \
      --fragment-detail-top-pairs 200 \
      --cell-line "$CELL_LINE" \
      --harmful-fragment-top-n 200 \
      --outdir "$RUN_ROOT_OUTPUT_CLEAN/shap"

  # ── Step 6: Symbolic regression ──────────────────────────────────────────
  run_step "6-SymReg" \
    python symbolic_regression_bioisostere_updated.py \
      --checkpoint "$CHECKPOINT" \
      --dataset "$DATASET" \
      --molecule-embeddings "$MOL_EMBS" \
      --fragment-embeddings "$FRAG_EMBS" \
      --fragment-map "$FRAG_MAP" \
      --drug-smiles-map "$DRUG_MAP" \
      --cell-line "$CELL_LINE" \
      --drop-source-features \
      --feature-top-k 15 \
      --top-fragments-table "$SHAP_OUT/top_200_harmful_fragments.csv" \
      --seed-selection-mode most_harmful \
      --harmful-seed-threshold 0.0 \
      --max-fragment-seeds 0 \
      --max-candidates-per-fragment 0 \
      --outdir "$RUN_ROOT_OUTPUT_CLEAN/symbolic"

  # ── Step 7: Annotate candidates with SDF paths ───────────────────────────
  run_step "7-Annotate" \
    python annotate_replacement_fragment_sdf.py \
      "$RANKED_CSV" \
      --smiles-dir "$SMILES_DIR" \
      --fragment-dir "$FRAG_DIR" \
      --output-csv "$WITH_SDF_CSV" \
      --top-output-csv "$TOP_CSV" \
      --score-column "symbolic_pred_delta_synergy" \
      --top-n 200 \
      --top-per-pair 5

  # ── Step 8: Filter compact CSV for Docker reconstruction ─────────────────
  run_step "8-Filter" \
    python filter_marked_smiles.py \
      "$TOP_CSV" \
      "$COMPACT_CSV"

  # ── Step 8.5: Docker graph-based reconstruction (eSynth) ─────────────────
  #
  #  NOTE: Docker maps the cell-line symbolic directory into /data.
  #        The scored output file is written INSIDE that same directory
  #        by Docker, so we chown it afterwards to avoid permission issues.
  #
  if [[ $CL_OK -eq 1 ]]; then
    log "  → Step 8.5-Docker (eSynth reconstruction)"
    SYM_ABS="$(realpath "$SYM_OUT")"
    if docker run --rm \
        -v "${SYM_ABS}:/data" \
        frag-replace \
        /data/replacement_candidates_top200_positive_with_fragment_sdf.csv \
        /data/replacement_candidates_top200_positive_with_fragment_sdf_scored.csv \
        --force-field UFF; then
      # Fix ownership (Docker writes files as root inside the container)
      sudo chown -R "$(id -un):$(id -gn)" "$SYM_ABS"
    else
      fail "Step 8.5-Docker failed for $CELL_LINE — skipping remaining steps."
      CL_OK=0
    fi
  fi

  # ── Step 9: Rescore reconstructed molecules ───────────────────────────────
  run_step "9-Rescore" \
    python rescore_reconstructed_candidates.py \
      "$SCORED_CSV" \
      --context-csv "$WITH_SDF_CSV" \
      --checkpoint "$CHECKPOINT" \
      --dataset "$DATASET" \
      --molecule-embeddings "$MOL_EMBS" \
      --fragment-embeddings "$FRAG_EMBS" \
      --fragment-map "$FRAG_MAP" \
      --drug-smiles-map "$DRUG_MAP" \
      --molecule-encoder-checkpoint "$MOL_ENC" \
      --fragment-encoder-checkpoint "$FRAG_ENC" \
      --cell-line "$CELL_LINE" \
      --outdir "$RESCORE_OUT"

  # ── Step 10: Drug repurposing ─────────────────────────────────────────────
  run_step "10-Repurposing" \
    python drug_repurposing.py \
      --rescored "$RESCORE_OUT/rescored_reconstructed_candidates.csv" \
      --frag-map "$FRAG_MAP" \
      --drug-map "$DRUG_MAP" \
      --moa-cache "$MOA_CACHE" \
      --outdir "$REPURPOSE_OUT" \
      --cell-line "$CELL_LINE" \
      --min-delta 5.0 \
      --exact-only \
      --plot-top-n 20

  # ── Result ────────────────────────────────────────────────────────────────
  if [[ $CL_OK -eq 1 ]]; then
    log "  ✓ DONE  $CELL_LINE"
    (( PASS++ )) || true
  else
    log "  ✗ FAILED  $CELL_LINE"
    FAIL_LIST+=("$CELL_LINE")
    # Write a marker file so you can grep easily later
    mkdir -p "$RUN_ROOT_OUTPUT_CLEAN/logs"
    echo "FAILED $(date)" > "$RUN_ROOT_OUTPUT_CLEAN/logs/${CELL_LINE}.FAILED"
  fi

done

# ── Final summary ─────────────────────────────────────────────────────────────
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "PIPELINE COMPLETE"
log "  Passed : $PASS / $TOTAL"
log "  Failed : ${#FAIL_LIST[@]} — ${FAIL_LIST[*]:-none}"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


##
# For every repurposing hit (replaced drug → repurpose drug), it does two independent checks:

# Check A — Target match (biological level)

# replaced_drug.target_chembl_ids  ∩  repurpose_drug.target_chembl_ids  ≠ ∅
# Both drugs inhibit/activate the exact same protein in ChEMBL. 
# This is the strongest possible match — the repurposing drug literally does the same molecular job.