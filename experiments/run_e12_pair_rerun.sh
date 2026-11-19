#!/usr/bin/env bash
# =============================================================================
#  E12b -- the unseen-pair split, rerun properly.
#
#  In the first E12 sweep two of four pair-split seeds never trained: best
#  epoch 2 and 3, train PCC 0.49 and 0.44, stopped by patience. Reporting a
#  mean over those is not defensible.
#
#  Rather than replace only the failures, which would be cherry-picking, this
#  runs the full four seeds under two uniform configurations:
#
#    orig    identical to the first sweep (encoder LR 1e-4, patience 40)
#            -> characterises the collapse rate, which is itself a result
#    stable  encoder LR 3e-5, patience 60
#            -> the estimate we report, if it trains reliably
#
#  8 runs, one wave on 8 GPUs, ~40 min.
#
#  Usage:  bash experiments/run_e12_pair_rerun.sh [n_parallel]
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
PY=/home/magesh/anaconda3/envs/molegen/bin/python
NPAR=${1:-8}
NGPU=$(nvidia-smi --list-gpus | wc -l)
LOG=experiments/results/logs
mkdir -p "$LOG"

JOBS=$(mktemp)
for s in 42 1 2 3; do
  echo "$PY experiments/e12_end2end_teacher.py --group-mode pair --seed $s --encoder-lr 1e-4 --patience 40 --outdir experiments/results/e2e_pair_orig_s${s} > $LOG/e2e_pair_orig_s${s}.log 2>&1" >> "$JOBS"
  echo "$PY experiments/e12_end2end_teacher.py --group-mode pair --seed $s --encoder-lr 3e-5 --patience 60 --outdir experiments/results/e2e_pair_stable_s${s} > $LOG/e2e_pair_stable_s${s}.log 2>&1" >> "$JOBS"
done

echo "queued $(wc -l < "$JOBS") runs, $NPAR at a time across $NGPU GPUs"
nl -ba "$JOBS" | while read -r i cmd; do
  echo "CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) $cmd"
done | xargs -P "$NPAR" -I CMD bash -c CMD
rm -f "$JOBS"

echo
echo "=== collapse audit ==="
$PY - <<'PY'
import json
from pathlib import Path
import numpy as np, pandas as pd
R = Path("experiments/results")
for cfg in ("orig", "stable"):
    rows = []
    for s in (42, 1, 2, 3):
        d = R / f"e2e_pair_{cfg}_s{s}"
        if not (d / "metrics.json").exists():
            continue
        m = json.load(open(d / "metrics.json"))
        h = pd.read_csv(d / "history.csv")
        best = int(h["val_rmse"].idxmin()) + 1
        rows.append((s, best, m["metrics_train"]["pearson"], m["metrics_test"]["pearson"]))
    if not rows:
        continue
    ok = [r for r in rows if r[1] > 10 and r[2] > 0.7]
    print(f"\n{cfg}:")
    for s, b, tr, te in rows:
        flag = "" if (b > 10 and tr > 0.7) else "   <- collapsed"
        print(f"  seed {s}: best_ep={b:3d} train={tr:.3f} test={te:.3f}{flag}")
    print(f"  collapsed {len(rows)-len(ok)}/{len(rows)}")
    if ok:
        v = [r[3] for r in ok]
        print(f"  healthy-seed test PCC: {np.mean(v):.4f} +/- "
              f"{np.std(v, ddof=1) if len(v) > 1 else 0.0:.4f}  (n={len(v)})")
PY
