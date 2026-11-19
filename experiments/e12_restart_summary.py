"""
E12 (restart summary) -- the learned encoder given four attempts per partition
==============================================================================
Reports the end-to-end encoder under a deliberately generous protocol: four
initialisations on every unseen-pair partition, scored at its best. The
comparison arms get one run per partition, so any remaining gap is an
underestimate of the true one.

Also reports the collapse rate, which is a result in its own right: how often a
training run of each representation fails to fit at all.

    python experiments/e12_restart_summary.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "experiments/results"
PARTS = (42, 1, 2, 3)
INITS = (0, 1, 2, 7)


def get(tag):
    p = R / tag / "metrics.json"
    if not p.exists():
        return None
    m = json.load(open(p))
    return m["metrics_train"]["pearson"], m["metrics_test"]["pearson"]


def single(pat, part):
    """One run per partition for the fixed-representation arms."""
    tag = pat if part == 42 else f"{pat}_s{part}"
    return get(tag)


def main():
    print("=" * 74)
    print("  UNSEEN-PAIR SPLIT: learned encoder with restarts vs the controls")
    print("=" * 74)

    rows, collapse = [], {}

    best_of, mean_of = [], []
    n_col = n_tot = 0
    print("\nend-to-end GCN, four initialisations per partition")
    for part in PARTS:
        vals, cols = [], 0
        for init in INITS:
            r = get(f"e2e_pair_p{part}_i{init}")
            if r is None:
                continue
            tr, te = r
            vals.append(te)
            if tr < 0.7:
                cols += 1
        if not vals:
            print(f"  partition {part}: no runs")
            continue
        n_col += cols
        n_tot += len(vals)
        best_of.append(max(vals))
        mean_of.append(np.mean(vals))
        print(f"  partition {part:2d}: best={max(vals):.3f}  mean={np.mean(vals):.3f}"
              f"  collapsed {cols}/{len(vals)}   [" +
              " ".join(f"{v:.3f}" for v in vals) + "]")
    if best_of:
        collapse["end-to-end GCN"] = (n_col, n_tot)
        rows.append(("end-to-end GCN (best of 4)", np.mean(best_of), np.std(best_of, ddof=1)))
        rows.append(("end-to-end GCN (mean of 4)", np.mean(mean_of), np.std(mean_of, ddof=1)))

    print("\ncontrols, one run per partition")
    for label, pat in (("one-hot drug identity", "onehot_drug_pair"),
                       ("frozen autoencoder", "onehot_none_pair"),
                       ("ECFP6 + descriptors", "ecfp_pair")):
        vals, cols = [], 0
        for part in PARTS:
            r = single(pat, part) if not pat.startswith("ecfp") else get(f"ecfp_pair_s{part}")
            if r is None:
                continue
            tr, te = r
            vals.append(te)
            if tr < 0.7:
                cols += 1
        if not vals:
            continue
        collapse[label] = (cols, len(vals))
        rows.append((label, np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
        print(f"  {label:<26} mean={np.mean(vals):.3f}  collapsed {cols}/{len(vals)}   ["
              + " ".join(f"{v:.3f}" for v in vals) + "]")

    print("\n" + "-" * 74)
    print(f"  {'representation':<30}{'mean PCC':>10}{'sd':>8}")
    print("  " + "-" * 48)
    for name, m, s in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<30}{m:>10.4f}{s:>8.4f}")

    print("\n  collapse rate (train PCC < 0.7)")
    for k, (c, n) in collapse.items():
        print(f"    {k:<28} {c}/{n}")

    pd.DataFrame(rows, columns=["representation", "mean_pcc", "sd"]).to_csv(
        R / "e12_restart_summary.csv", index=False)
    print("\n  -> experiments/results/e12_restart_summary.csv")


if __name__ == "__main__":
    main()
