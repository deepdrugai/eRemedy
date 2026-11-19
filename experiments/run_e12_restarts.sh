#!/usr/bin/env bash
# =============================================================================
#  E12c -- is the collapse a property of the partition or of the initialisation?
#
#  On pair-split seeds 2 and 3 the end-to-end encoder fails: validation PCC
#  peaks near 0.35 and never recovers, while one-hot reaches 0.63-0.76 on the
#  same partitions. Lowering the encoder LR did not help, and the histories
#  show this is not an early-stopping artefact -- the model genuinely does not
#  generalise there.
#
#  But --seed previously fixed the partition AND the initialisation together,
#  so the two are confounded. This holds the partition fixed on the two failing
#  seeds and varies the initialisation four ways.
#
#    all inits fail   -> the representation cannot fit those partitions.
#                        A real finding, reported as-is.
#    some inits work  -> optimisation fragility, fixable by restarts. Then the
#                        same restart budget must be given to every arm.
#
#  Also completes the ECFP grid, which is missing seed 3 on both splits.
#
#  Usage:  bash experiments/run_e12_restarts.sh [n_parallel]
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
PY=/home/magesh/anaconda3/envs/molegen/bin/python
NPAR=${1:-8}
NGPU=$(nvidia-smi --list-gpus | wc -l)
LOG=experiments/results/logs
ECFP=clean_runs/2026-08-28_ecfp_rerun/ecfp
FRAGMAP=clean_runs/2026-04-29_full_rerun/fragment_ae/smiles_to_fragments.pkl
mkdir -p "$LOG"

JOBS=$(mktemp)

# restarts on the two failing partitions
for part in 2 3; do
  for init in 0 1 2 7; do
    tag="e2e_pair_p${part}_i${init}"
    echo "$PY experiments/e12_end2end_teacher.py --group-mode pair --seed $part --init-seed $init --outdir experiments/results/$tag > $LOG/$tag.log 2>&1" >> "$JOBS"
  done
done

# complete the ECFP grid (seed 3, both splits)
for sp_ in cellpair pair; do
  tag="ecfp_${sp_}_s3"
  echo "$PY experiments/e8_onehot_controls.py --ablate none --group-mode $sp_ --seed 3 --molecule-embeddings $ECFP/molecule_smiles_embeddings.pkl --fragment-embeddings $ECFP/fragment_smiles_embeddings.pkl --fragment-map $FRAGMAP --outdir experiments/results/$tag > $LOG/$tag.log 2>&1" >> "$JOBS"
done

echo "queued $(wc -l < "$JOBS") runs, $NPAR at a time across $NGPU GPUs"
nl -ba "$JOBS" | while read -r i cmd; do
  echo "CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) $cmd"
done | xargs -P "$NPAR" -I CMD bash -c CMD
rm -f "$JOBS"

echo
echo "=== restart audit: does any initialisation rescue the partition? ==="
$PY - <<'PY'
import json
from pathlib import Path
import numpy as np
R = Path("experiments/results")
onehot = {2: 0.627, 3: 0.756}
for part in (2, 3):
    vals = []
    print(f"\npartition seed {part}   (one-hot on this partition: {onehot[part]:.3f})")
    for init in (0, 1, 2, 7):
        p = R / f"e2e_pair_p{part}_i{init}" / "metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        tr, te = m["metrics_train"]["pearson"], m["metrics_test"]["pearson"]
        vals.append(te)
        print(f"  init {init}: train={tr:.3f}  test={te:.3f}"
              f"{'' if tr > 0.7 else '   <- still collapsed'}")
    if vals:
        print(f"  best over restarts: {max(vals):.3f} | mean {np.mean(vals):.3f}")
        print("  -> %s" % ("RESCUED by restarts" if max(vals) > 0.55
                           else "not rescued; partition-level failure"))
PY
