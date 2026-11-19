"""
E13 (aggregate) -- the vocabulary threshold curve
=================================================
Plots test PCC against vocabulary size for the three drug representations, and
underneath it the quantity the claim actually rests on: learned minus one-hot.
Where that crosses zero is the threshold above which learning a molecular
representation starts to pay for itself.

    python experiments/e13_plot_scaling.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments/results"
FIG = REPO / "thesis/figures"

SIZES = (21, 30, 40, 50, 60)
SEEDS = (42, 1, 2)
ARMS = [("learned", "End-to-end GCN (learned)", "#1a5fb4", "-o"),
        ("onehot", "One-hot drug identity", "#c0392b", "--^"),
        ("ecfp", "ECFP6 + descriptors (fixed)", "#1a7f37", "-s")]


def collect(prefix):
    rows = []
    for arm, _, _, _ in ARMS:
        for V in SIZES:
            for s in SEEDS:
                p = RES / f"{prefix}_{arm}_v{V}_s{s}" / "metrics.json"
                if not p.exists():
                    continue
                d = json.load(open(p))
                a = json.load(open(RES / f"{prefix}_{arm}_v{V}_s{s}" / "run_audit.json"))
                rows.append(dict(arm=arm, n_drugs=V, seed=s,
                                 rows_used=a.get("rows_used"),
                                 test=d["metrics_test"]["pearson"],
                                 train=d["metrics_train"]["pearson"]))
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="scale")
    ap.add_argument("--subtitle", default="training rows held constant at 8,500")
    ap.add_argument("--out", default="vocabulary_threshold")
    ap.add_argument("--no-title", action="store_true",
                    help="omit the in-figure title; use for the paper, where "
                         "the caption carries that information")
    A = ap.parse_args()
    globals()["_A"] = A

    D = collect(A.prefix)
    if D.empty:
        raise SystemExit("no scaling runs found yet")
    D.to_csv(RES / ("e13_%s_raw.csv" % A.prefix), index=False)

    g = D.groupby(["arm", "n_drugs"]).test.agg(["mean", "std", "count"]).reset_index()
    print("=" * 66)
    print("  E13  VOCABULARY SCALING ON NCI-ALMANAC (%s)" % A.subtitle)
    print("=" * 66)
    for arm, label, _, _ in ARMS:
        sub = g[g.arm == arm]
        if sub.empty:
            continue
        print("\n  %s" % label)
        for _, r in sub.iterrows():
            sd = 0.0 if np.isnan(r["std"]) else r["std"]
            print("    n_drugs=%2d  PCC=%.4f +/- %.4f  (n=%d)"
                  % (r.n_drugs, r["mean"], sd, r["count"]))

    piv = D.pivot_table(index="n_drugs", columns="arm", values="test", aggfunc="mean")
    delta = None
    if {"learned", "onehot"}.issubset(piv.columns):
        delta = (piv["learned"] - piv["onehot"]).reindex(SIZES)
        print("\n  learned minus one-hot")
        for V in SIZES:
            if V in delta.index and np.isfinite(delta.loc[V]):
                print("    n_drugs=%2d  %+.4f" % (V, delta.loc[V]))
        cross = [V for V in SIZES if V in delta.index and delta.loc[V] > 0]
        print("\n  crosses zero at n_drugs >= %s"
              % (min(cross) if cross else "never in this range"))

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.1, 1]))
    for arm, label, colour, style in ARMS:
        sub = g[g.arm == arm].sort_values("n_drugs")
        if sub.empty:
            continue
        sd = sub["std"].fillna(0.0).values
        ax.errorbar(sub.n_drugs, sub["mean"], yerr=sd, fmt=style, color=colour,
                    lw=2.2, ms=8, capsize=4, label=label)
    ax.axvline(21, color="#8a8a8a", ls=":", lw=1.5)
    ax.text(21.4, ax.get_ylim()[0] + 0.02, "O'Neil\nvocabulary", fontsize=9,
            color="#8a8a8a", va="bottom")
    ax.set_ylabel("Test Pearson correlation", fontsize=12)
    if not A.no_title:
        ax.set_title("Does a learned molecular representation need a vocabulary?\n"
                     "NCI-ALMANAC, %s" % A.subtitle,
                     fontsize=13, fontweight="bold")
    ax.legend(fontsize=10.5, loc="best")
    ax.grid(alpha=0.25)

    if delta is not None:
        d = delta.dropna()
        bx.axhspan(-1, 0, color="#c0392b", alpha=0.06)
        bx.axhline(0, color="black", lw=1.2)
        bx.plot(d.index, d.values, "-o", color="#1a5fb4", lw=2.4, ms=9)
        for V, v in d.items():
            bx.annotate("%+.3f" % v, (V, v), xytext=(0, 11 if v >= 0 else -18),
                        textcoords="offset points", ha="center", fontsize=9,
                        color="#1a5fb4")
        lo, hi = float(d.min()), float(d.max())
        pad = max(0.05, 0.25 * (hi - lo))
        bx.set_ylim(lo - pad, hi + pad)
        bx.axvline(21, color="#8a8a8a", ls=":", lw=1.5)
        bx.set_ylabel("learned $-$ one-hot", fontsize=11)
        bx.grid(alpha=0.25)
    bx.set_xlabel("number of distinct drugs in the training vocabulary",
                  fontsize=12)
    bx.set_xticks(list(SIZES))

    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(FIG / ("%s.%s" % (A.out, ext)), dpi=200,
                    bbox_inches="tight")
    plt.close()
    print("\n  written -> thesis/figures/%s.{pdf,png}" % A.out)


if __name__ == "__main__":
    main()
