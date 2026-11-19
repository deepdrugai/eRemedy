"""
gen_figures_shap_divergence.py
==============================
Two figures built directly from the run outputs:

  1. modality_shap.pdf            pooled modality-level SHAP across 34 cell lines
  2. cross_cellline_divergence.pdf  sign divergence of fragment swaps across lines
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = "clean_runs/clean_runs/2026-07-10_full_rerun_inference"
FIG = "thesis/figures"
BLUE, GREEN, RED, AMBER, GREY = "#1a5fb4", "#1a7f37", "#c0392b", "#b8860b", "#666666"

LABEL = {
    "cell": "cell context",
    "graph_total": "whole-molecule (both drugs)",
    "fragment_total": "fragments (both drugs)",
    "drug_a_graph": "drug A whole-molecule",
    "drug_b_graph": "drug B whole-molecule",
    "drug_a_fragments": "drug A fragments",
    "drug_b_fragments": "drug B fragments",
}


def save(name):
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, name + "." + ext), dpi=200, bbox_inches="tight")
    plt.close()
    print("  ->", name)


def modality_shap():
    rows = []
    for f in sorted(glob.glob(os.path.join(RUN, "shap/*/modality_shap_summary.csv"))):
        d = pd.read_csv(f)
        d["cell_line"] = f.split(os.sep)[-2]
        rows.append(d)
    m = pd.concat(rows, ignore_index=True)
    piv = m.pivot_table(index="cell_line", columns="feature", values="mean_abs_shap")

    order = ["cell", "fragment_total", "graph_total",
             "drug_a_fragments", "drug_b_graph", "drug_a_graph", "drug_b_fragments"]
    order = [o for o in order if o in piv.columns]

    # Triangle layout: the per-channel distribution spans the top, the two
    # derived panels sit below it. "Channel" throughout, since "modality"
    # already denotes the six omics modalities of the data section.
    fig = plt.figure(figsize=(13.0, 10.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0],
                          hspace=0.42, wspace=0.26)
    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    cols = [RED, BLUE, GREEN, GREY, GREY, GREY, GREY]

    # top: distribution per channel across the 34 cell lines
    data = [piv[o].dropna().values for o in order]
    bp = ax_top.boxplot(data, vert=False, patch_artist=True, widths=0.64)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(2.0)
    ax_top.set_yticklabels([LABEL.get(o, o) for o in order], fontsize=13)
    ax_top.set_xlabel("mean |SHAP| contribution", fontsize=14, color="black")
    ax_top.set_title("Channel importance across 34 cell lines",
                     fontsize=15, fontweight="bold", color="black")

    # bottom left: fragment vs whole-molecule, per cell line
    ax_bl.scatter(piv.graph_total, piv.fragment_total, s=90, c=BLUE,
                  edgecolor="black", linewidth=0.9, alpha=0.85)
    lim = [min(piv.graph_total.min(), piv.fragment_total.min()) - 1,
           max(piv.graph_total.max(), piv.fragment_total.max()) + 1]
    ax_bl.plot(lim, lim, "k--", lw=1.4)
    n_frag = int((piv.fragment_total > piv.graph_total).sum())
    ax_bl.set_xlabel("whole-molecule mean |SHAP|", fontsize=14, color="black")
    ax_bl.set_ylabel("fragment mean |SHAP|", fontsize=14, color="black")
    ax_bl.set_title("Fragment vs whole-molecule channel\n"
                    "(%d of %d lines above the diagonal)" % (n_frag, len(piv)),
                    fontsize=15, fontweight="bold", color="black")

    # bottom right: spread, which channel varies most
    sd = m.pivot_table(index="cell_line", columns="feature", values="std_shap")
    vals = [sd[o].mean() for o in order]
    ax_br.barh(range(len(order)), vals, color=cols[:len(order)],
               edgecolor="black", linewidth=1.0)
    ax_br.set_yticks(range(len(order)))
    ax_br.set_yticklabels([LABEL.get(o, o) for o in order], fontsize=13)
    ax_br.yaxis.tick_right()          # keep labels clear of the left-hand panel
    ax_br.invert_yaxis()
    ax_br.set_xlabel("mean within-line SHAP std. dev.", fontsize=14, color="black")
    ax_br.set_title("Which channel varies most\nbetween drug pairs",
                    fontsize=15, fontweight="bold", color="black")

    for a in (ax_top, ax_bl, ax_br):
        a.tick_params(colors="black", labelsize=12)
        for s in a.spines.values():
            s.set_color("black")
    save("modality_shap")

    print("     cell largest in %d/%d lines" %
          (int((piv[["cell", "graph_total", "fragment_total"]].idxmax(axis=1) == "cell").sum()),
           len(piv)))


def divergence(min_lines=15):
    r = pd.read_csv("thesis/aggregate/rescored_pooled.csv")
    sw = (r.groupby(["source_fragment_smiles", "candidate_fragment_smiles",
                     "cell_line_name"]).rescored_delta_synergy_vs_model
            .mean().reset_index())
    a = (sw.groupby(["source_fragment_smiles", "candidate_fragment_smiles"])
           .rescored_delta_synergy_vs_model
           .agg(lines="count", mean="mean", mn="min", mx="max",
                npos=lambda s: (s > 0).sum(), nneg=lambda s: (s < 0).sum())
           .reset_index())
    a = a[a.lines >= min_lines].copy()
    a["spread"] = a.mx - a.mn
    a = a.sort_values("spread", ascending=False)

    fig, ax = plt.subplots(1, 2, figsize=(15.0, 6.0))

    t = a.head(12).iloc[::-1]
    y = np.arange(len(t))
    ax[0].hlines(y, t.mn, t.mx, color=GREY, lw=2.4, zorder=1)
    ax[0].scatter(t.mn, y, s=76, c=RED, edgecolor="black", zorder=3,
                  label="worst cell line")
    ax[0].scatter(t.mx, y, s=76, c=GREEN, edgecolor="black", zorder=3,
                  label="best cell line")
    ax[0].scatter(t["mean"], y, s=52, marker="D", c=BLUE, edgecolor="black",
                  zorder=4, label="mean")
    ax[0].axvline(0, color="black", lw=1.4, ls="--")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(["%s\n$\\rightarrow$ %s  (%d lines)"
                           % (s[:22], c[:18], n)
                           for s, c, n in zip(t.source_fragment_smiles,
                                              t.candidate_fragment_smiles, t.lines)],
                          fontsize=7.5)
    ax[0].set_xlabel("rescored synergy change (Loewe)", fontsize=11, color="black")
    ax[0].set_title("The same swap, opposite outcomes\nacross cell lines",
                    fontsize=12, fontweight="bold", color="black")
    ax[0].legend(fontsize=9, loc="lower right")

    div = ((a.npos > 0) & (a.nneg > 0)).sum()
    ax[1].bar(["consistent\nin sign", "sign-divergent\nacross lines"],
              [len(a) - div, div], color=[GREEN, RED],
              edgecolor="black", linewidth=1.2)
    for i, v in enumerate([len(a) - div, div]):
        ax[1].text(i, v + 0.6, "%d\n(%.0f%%)" % (v, 100 * v / len(a)),
                   ha="center", fontsize=11, fontweight="bold", color="black")
    ax[1].set_ylabel("fragment swaps recovered in $\\geq$%d cell lines" % min_lines,
                     fontsize=11, color="black")
    ax[1].set_ylim(0, len(a) * 1.18)
    ax[1].set_title("Most swaps are not universally beneficial",
                    fontsize=12, fontweight="bold", color="black")

    for x in ax:
        x.tick_params(colors="black", labelsize=9)
        for s in x.spines.values():
            s.set_color("black")
    plt.tight_layout()
    save("cross_cellline_divergence")

    print("     %d swaps in >=%d lines, %d sign-divergent (%.0f%%)"
          % (len(a), min_lines, div, 100 * div / len(a)))


if __name__ == "__main__":
    print("generating figures ...")
    modality_shap()
    divergence()
    print("done")
