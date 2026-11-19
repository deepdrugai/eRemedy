"""
Figure: the generalisation ladder
=================================
Five evaluation regimes, ordered by how much novel chemistry each demands,
with the teacher, DeepSynergy, a gradient-boosting baseline on the same
molecular embeddings, and a structure-free lookup table side by side.

The point of the figure is that the curves cross, and that every learned model
declines along the same ladder regardless of capacity. A lookup table wins
where the split leaks and collapses everywhere else; DeepSynergy leads at every
rung it can be evaluated on; and everything fails once the drug itself is
unseen. Teacher error bars are +/-1 SD over four seeds, which is what shows the
unseen-pair split cannot support a point estimate.

Writes thesis/figures/generalisation_ladder.{pdf,png}. Read-only otherwise.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments/results"
FIG = REPO / "thesis/figures"

BLUE, GREEN, RED, GREY = "#1a5fb4", "#1a7f37", "#c0392b", "#8a8a8a"
PURPLE = "#7b3fa0"

REGIMES = [
    ("Random split",                 "same combination\nin train and test"),
    ("Unseen\ncombination",          "known drugs,\nnew pairing"),
    ("Unseen\ndrug pair",            "pair absent from\nevery cell line"),
    ("Unseen drug\n(LODO)",          "drug held out\nentirely"),
    ("Unseen drug,\nnew study",      "NCI-ALMANAC\ntransfer"),
]


def pcc(rel):
    p = RES / rel
    return json.load(open(p))["metrics_test"]["pearson"] if p.exists() else np.nan


def seed_stats(split):
    """Teacher across four seeds under a grouped split."""
    tags = ["onehot_none_%s" % split] + \
           ["onehot_none_%s_s%d" % (split, s) for s in (1, 2, 3)]
    v = [pcc("%s/metrics.json" % t) for t in tags]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def load():
    cp_m, cp_sd, cp_n = seed_stats("cellpair")
    pr_m, pr_sd, pr_n = seed_stats("pair")

    teacher = [0.9463, cp_m, pr_m, 0.285, 0.043]
    teacher_sd = [0.0, cp_sd, pr_sd, 0.0, 0.0]

    deepsyn = [pcc("deepsynergy_random/metrics.json"),
               pcc("deepsynergy_cellpair/metrics.json"),
               pcc("deepsynergy_pair/metrics.json"),
               np.nan, np.nan]

    gbm_rows = {r["split_mode"]: r["pcc"]
                for r in json.load(open(RES / "e6_gbm_baseline.json"))}
    gbm = [gbm_rows["random"], gbm_rows["cellpair"], gbm_rows["pair"],
           np.nan, np.nan]

    # lookup: wins under leakage, has no entry to recall otherwise and backs
    # off to the cell-line mean (E3, vocabulary-restricted)
    lookup = [0.969, 0.183, 0.183, np.nan, np.nan]
    return teacher, teacher_sd, deepsyn, gbm, lookup, (cp_n, pr_n)


def main():
    os.makedirs(FIG, exist_ok=True)
    teacher, teacher_sd, deepsyn, gbm, lookup, ns = load()
    x = np.arange(len(REGIMES))

    fig, ax = plt.subplots(figsize=(11.5, 6.0))

    ax.axhspan(-0.05, 0.30, color=RED, alpha=0.06, zorder=0)
    ax.text(4.42, 0.13, "no usable\nsignal", ha="center", va="center",
            fontsize=9, color=RED, style="italic", zorder=1)

    ax.errorbar(x, teacher, yerr=teacher_sd, fmt="-o", color=BLUE, lw=2.6,
                ms=11, capsize=5, elinewidth=1.8, zorder=5,
                label="Teacher (multi-modal GNN, 0.94M)")
    ax.plot(x[:3], deepsyn[:3], "-D", color=PURPLE, lw=2.2, ms=8.5, zorder=4,
            label="DeepSynergy (86.5M)")
    ax.plot(x[:3], gbm[:3], "-s", color=GREEN, lw=2.2, ms=9, zorder=3,
            label="Gradient boosting, same embeddings")
    ax.plot(x[:3], lookup[:3], "--^", color=GREY, lw=2.0, ms=9, zorder=2,
            label="Lookup table (no structure)")

    for xi, v, sd in zip(x, teacher, teacher_sd):
        off = 13 + (26 if sd > 0.05 else 0)
        ax.annotate("%.3f" % v, (xi, v), xytext=(0, off),
                    textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=BLUE)
    for xi, v in zip(x[:3], deepsyn[:3]):
        ax.annotate("%.3f" % v, (xi, v), xytext=(0, 12),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=PURPLE)
    for xi, v in zip(x[:3], gbm[:3]):
        ax.annotate("%.3f" % v, (xi, v), xytext=(0, -19),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=GREEN)
    for xi, v in zip(x[:3], lookup[:3]):
        off = (-34, 6) if xi == 0 else (0, -19)
        ax.annotate("%.3f" % v, (xi, v), xytext=off,
                    textcoords="offset points", ha="center",
                    fontsize=9, color=GREY)

    # the unseen-pair split is the one that cannot carry a point estimate
    ax.annotate("single-run estimates here\nspan 0.26-0.73",
                xy=(2, teacher[2] - teacher_sd[2]), xytext=(2.45, 0.40),
                fontsize=9, color=BLUE, style="italic",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))

    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n\n%s" % (a, b) for a, b in REGIMES], fontsize=9.5)
    ax.set_ylabel("Test Pearson correlation", fontsize=12, color="black")
    ax.set_ylim(-0.05, 1.06)
    ax.set_xlim(-0.4, 4.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Accuracy against how much novel chemistry the task demands",
                 fontsize=13, fontweight="bold", color="black")
    ax.legend(fontsize=10.5, loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_color("black")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG / ("generalisation_ladder." + ext), dpi=200,
                    bbox_inches="tight")
    plt.close()
    print("written -> thesis/figures/generalisation_ladder.{pdf,png}")
    print()
    fmt = lambda v: " ".join("%.3f" % t if np.isfinite(t) else "  -  " for t in v)
    print("  teacher   :", fmt(teacher), "  (n seeds: cellpair %d, pair %d)" % ns)
    print("  +/- SD    :", fmt(teacher_sd))
    print("  deepsyn   :", fmt(deepsyn))
    print("  gbm       :", fmt(gbm))
    print("  lookup    :", fmt(lookup))


if __name__ == "__main__":
    main()
