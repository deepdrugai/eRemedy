#!/usr/bin/env bash
# =============================================================================
#  E11 -- full pipeline re-run with ECFP molecular featurisation
#
#  Motivation: E8 showed one-hot drug identity matches or beats the teacher's
#  autoencoder embeddings under grouped splits, while E10 showed DeepSynergy
#  loses 0.11-0.15 PCC when its ECFP features are swapped for the same one-hot
#  code. Chemistry is learnable on O'Neil; the SMILES-autoencoder embeddings
#  are what fail to carry it. This re-runs the whole chain on ECFP6 features so
#  the paper can report both architectures.
#
#  Only the featuriser changes. The fragment channel stays fragment-resolved,
#  so fragment SHAP, leave-one-out ranking and Flow 2 all work unchanged.
#
#  Usage
#  -----
#    bash experiments/run_e11_ecfp_pipeline.sh <phase> [n_parallel]
#
#  Phases (run in this order, or use "all" for teacher..aggregate):
#    embed       build the ECFP pickles                    (~1 min, CPU)
#    headtohead  ECFP vs autoencoder vs one-hot,
#                grouped splits, 3 seeds                   (GPU)
#    teacher     train the production ECFP teacher         (GPU)
#    shap        fragment SHAP, 34 cell lines              (GPU)
#    sr          symbolic regression, 34 cell lines        (CPU, gplearn)
#    flowb       Flow 2 mechanism-matched repurposing      (GPU)
#    aggregate   pool Flow 2 hits and finalise             (CPU)
#    all         teacher -> shap -> sr -> flowb -> aggregate
#
#  Outputs live under $RUN so nothing existing is overwritten.
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY=/home/magesh/anaconda3/envs/molegen/bin/python

RUN="$REPO/clean_runs/2026-08-28_ecfp_rerun"
RUN0="$REPO/clean_runs/2026-04-29_full_rerun"
DATASET="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl"

ECFP_DIR="$RUN/ecfp"
MOL_EMBS="$ECFP_DIR/molecule_smiles_embeddings.pkl"
FRAG_EMBS="$ECFP_DIR/fragment_smiles_embeddings.pkl"
FRAG_MAP="$RUN0/fragment_ae/smiles_to_fragments.pkl"
DRUG_MAP="$REPO/everything_drug_smiles_map.csv"
MOA_CACHE="$REPO/drug_moa_cache_full.csv"

SYNERGY_OUT="$RUN/synergy/oneil_synergy_runs"
CHECKPOINT="$SYNERGY_OUT/model.pt"
SHAP_ROOT="$RUN/shap"
SYM_ROOT="$RUN/symbolic"
FLOWB_ROOT="$RUN/flow_b_shap_moa"
AGG_DIR="$RUN/aggregate"
FIG_DIR="$RUN/figures"
LOG="$RUN/logs"

CELL_LINES=(
  A2058 A2780 A375 A427 CAOV3 DLD1 ES2 HCT116 HT144 HT29 KPL1 LOVO
  MDAMB436 MSTO NCIH1650 NCIH2122 NCIH23 NCI-H460 NCIH520 OV90 OVCAR3
  PA1 RKO RPMI7951 SKMEL30 SKMES1 SK-OV-3 SW-620 SW837 T-47D UACC62
  UWB1289 VCAP ZR751
)

PHASE="${1:-}"
NPAR="${2:-6}"
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l); NGPU=${NGPU:-1}
[[ $NGPU -lt 1 ]] && NGPU=1

mkdir -p "$LOG" "$AGG_DIR" "$FIG_DIR"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Run a newline-delimited job file, n at a time, round-robin over the GPUs.
fan_out() {
  local jobs="$1" gpu="${2:-yes}" n
  n=$(wc -l < "$jobs")
  [[ $n -eq 0 ]] && { log "nothing to do"; return 0; }
  log "launching $n jobs, $NPAR concurrent$( [[ $gpu == yes ]] && echo " across $NGPU GPUs")"
  if [[ $gpu == yes ]]; then
    nl -ba "$jobs" | while read -r i cmd; do
      echo "CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) $cmd"
    done | xargs -P "$NPAR" -I CMD bash -c CMD
  else
    xargs -P "$NPAR" -I CMD bash -c CMD < "$jobs"
  fi
}

require() {
  [[ -f "$1" ]] || { log "MISSING: $1"; log "run the earlier phase first"; exit 1; }
}

# ---------------------------------------------------------------- embed -----
phase_embed() {
  log "PHASE embed"
  $PY experiments/e11_build_ecfp_embeddings.py --outdir "$ECFP_DIR" \
    2>&1 | tee "$LOG/embed.log"
}

# ----------------------------------------------------------- headtohead -----
# Reuses the E8 harness, which already does leakage-free grouped splits and
# an audited train/val/test partition. --ablate none keeps the real
# embeddings, so pointing it at the ECFP pickles gives ECFP under exactly the
# protocol used for the autoencoder and one-hot arms.
phase_headtohead() {
  log "PHASE headtohead (ECFP under grouped splits, seeds 42/1/2)"
  require "$MOL_EMBS"
  local jobs; jobs=$(mktemp)
  for s in 42 1 2; do
    for split in cellpair pair; do
      local tag="ecfp_${split}_s${s}"
      echo "$PY experiments/e8_onehot_controls.py --ablate none --group-mode $split --seed $s --molecule-embeddings $MOL_EMBS --fragment-embeddings $FRAG_EMBS --fragment-map $FRAG_MAP --outdir experiments/results/$tag > $LOG/$tag.log 2>&1" >> "$jobs"
    done
  done
  fan_out "$jobs"; rm -f "$jobs"
  log "compare against onehot_none_* (autoencoder) and onehot_drug_* (one-hot)"
}

# -------------------------------------------------------------- teacher -----
phase_teacher() {
  log "PHASE teacher"
  require "$MOL_EMBS"
  mkdir -p "$SYNERGY_OUT"
  $PY synergy_prediction.py \
    --input "$DATASET" \
    --molecule-embeddings "$MOL_EMBS" \
    --fragment-embeddings "$FRAG_EMBS" \
    --fragment-map "$FRAG_MAP" \
    --drug-smiles-map "$DRUG_MAP" \
    --outdir "$SYNERGY_OUT" \
    2>&1 | tee "$LOG/teacher.log"
  require "$CHECKPOINT"
  log "checkpoint -> $CHECKPOINT"
}

# ----------------------------------------------------------------- shap -----
phase_shap() {
  log "PHASE shap (${#CELL_LINES[@]} cell lines)"
  require "$CHECKPOINT"
  local jobs; jobs=$(mktemp)
  for CL in "${CELL_LINES[@]}"; do
    echo "$PY synergy_shap_analysis.py --checkpoint $CHECKPOINT --dataset $DATASET --molecule-embeddings $MOL_EMBS --fragment-embeddings $FRAG_EMBS --fragment-map $FRAG_MAP --drug-smiles-map $DRUG_MAP --cell-line $CL --fragment-detail-top-pairs 200 --harmful-fragment-top-n 200 --outdir $SHAP_ROOT > $LOG/shap_${CL}.log 2>&1" >> "$jobs"
  done
  fan_out "$jobs"; rm -f "$jobs"
  log "SHAP tables present: $(ls -d "$SHAP_ROOT"/*/top_200_harmful_fragments.csv 2>/dev/null | wc -l)/${#CELL_LINES[@]}"
}

# ------------------------------------------------------------------- sr -----
phase_sr() {
  log "PHASE sr (gplearn, CPU)"
  local jobs; jobs=$(mktemp)
  for CL in "${CELL_LINES[@]}"; do
    local frags="$SHAP_ROOT/$CL/top_200_harmful_fragments.csv"
    [[ -f "$frags" ]] || { log "  skip $CL (no SHAP table)"; continue; }
    echo "$PY symbolic_regression_bioisostere_updated.py --checkpoint $CHECKPOINT --dataset $DATASET --molecule-embeddings $MOL_EMBS --fragment-embeddings $FRAG_EMBS --fragment-map $FRAG_MAP --drug-smiles-map $DRUG_MAP --cell-line $CL --top-fragments-table $frags --seed-selection-mode most_harmful --harmful-seed-threshold 0.0 --drop-source-features --feature-top-k 15 --max-fragment-seeds 0 --max-candidates-per-fragment 0 --outdir $SYM_ROOT > $LOG/sr_${CL}.log 2>&1" >> "$jobs"
  done
  fan_out "$jobs" no; rm -f "$jobs"
  log "SR outputs: $(ls -d "$SYM_ROOT"/*/replacement_candidates_ranked.csv 2>/dev/null | wc -l)"
}

# ---------------------------------------------------------------- flowb -----
phase_flowb() {
  log "PHASE flowb (Flow 2, mechanism-matched, teacher as oracle)"
  require "$CHECKPOINT"
  local jobs; jobs=$(mktemp)
  for CL in "${CELL_LINES[@]}"; do
    local frags="$SHAP_ROOT/$CL/top_200_harmful_fragments.csv"
    [[ -f "$frags" ]] || { log "  skip $CL (no SHAP table)"; continue; }
    echo "$PY flow_b_shap_moa.py --shap-harmful $frags --checkpoint $CHECKPOINT --dataset $DATASET --cell-line $CL --outdir $FLOWB_ROOT/$CL --moa-cache $MOA_CACHE --molecule-embeddings $MOL_EMBS --fragment-embeddings $FRAG_EMBS --fragment-map $FRAG_MAP --drug-smiles-map $DRUG_MAP --top-fragments 50 > $LOG/flowb_${CL}.log 2>&1" >> "$jobs"
  done
  fan_out "$jobs"; rm -f "$jobs"
  log "Flow B hit files: $(ls "$FLOWB_ROOT"/*/flowB_shap_moa_hits_*.csv 2>/dev/null | wc -l)"
}

# ------------------------------------------------------------ aggregate -----
phase_aggregate() {
  log "PHASE aggregate"
  $PY aggregate_flow_b_shap_moa.py \
    --root "$FLOWB_ROOT" --outdir "$AGG_DIR" --figdir "$FIG_DIR" \
    2>&1 | tee "$LOG/aggregate.log"
  $PY finalize_flow_b.py \
    --hits "$AGG_DIR/flow_b_shap_moa_all_hits.csv" \
    --all-scores "$FLOWB_ROOT/*/flowB_shap_moa_all_*.csv" \
    --outdir "$AGG_DIR" --figdir "$FIG_DIR" \
    2>&1 | tee "$LOG/finalize.log"
  log "aggregate -> $AGG_DIR"
}

case "$PHASE" in
  embed)      phase_embed ;;
  headtohead) phase_headtohead ;;
  teacher)    phase_teacher ;;
  shap)       phase_shap ;;
  sr)         phase_sr ;;
  flowb)      phase_flowb ;;
  aggregate)  phase_aggregate ;;
  all)        phase_teacher && phase_shap && phase_sr && phase_flowb && phase_aggregate ;;
  *)
    sed -n '2,32p' "$0"
    exit 1 ;;
esac

log "phase '$PHASE' complete"
