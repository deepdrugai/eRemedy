"""
Generate figures for the pan-cancer (34 cell line) results chapter.
Run:  python gen_figures_pancancer.py
Outputs: figures/pc_*.svg and .pdf
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.join(HERE, "aggregate")
OUTDIR = os.path.join(HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#333333",
    "savefig.bbox": "tight",
})

TISSUE_COLOR = {
    "skin":            "#D62728",
    "lung":            "#1F77B4",
    "large_intestine": "#2CA02C",
    "ovary":           "#9467BD",
    "breast":          "#E377C2",
    "prostate":        "#FF7F0E",
}

sym = pd.read_csv(os.path.join(AGG, "symbolic_metrics.csv"))
res = pd.read_csv(os.path.join(AGG, "rescored_summary.csv"))
tis = pd.read_csv(os.path.join(AGG, "tissue_summary.csv"))
master = pd.read_csv(os.path.join(AGG, "master_table.csv"))
pooled = pd.read_csv(os.path.join(AGG, "rescored_pooled.csv"))
hi = pd.read_csv(os.path.join(AGG, "moa_high_pairings_fixed.csv"))
per_moa = pd.read_csv(os.path.join(AGG, "repurposing_moa_per_cellline_fixed.csv"))
xfrag = pd.read_csv(os.path.join(AGG, "cross_cellline_fragments.csv"))


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, "%s.%s" % (name, ext)), dpi=200)
    plt.close(fig)
    print("  wrote", name)


# ── F1. Symbolic model quality across 34 cell lines ─────────────────────────
def fig_symbolic_quality():
    d = sym.sort_values("test_pcc", ascending=True)
    colors = [TISSUE_COLOR.get(t, "#888888") for t in d.tissue]
    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(d))
    ax.barh(y, d.test_pcc, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d.cell_line, fontsize=8.5)
    ax.set_xlabel("Test-set Pearson correlation coefficient (PCC)", fontweight="bold")
    ax.set_title("Symbolic surrogate model quality across 34 cell lines",
                 fontweight="bold", fontsize=11)
    ax.axvline(0.5, color="#444444", ls="--", lw=1.0)
    ax.axvline(0.3, color="#888888", ls=":", lw=1.0)
    ax.text(0.505, 1.0, "strong (≥0.50)", fontsize=7.5, rotation=90, va="bottom")
    ax.text(0.305, 1.0, "moderate (≥0.30)", fontsize=7.5, rotation=90, va="bottom")
    for i, v in enumerate(d.test_pcc):
        ax.text(v + 0.008, i, "%.2f" % v, va="center", fontsize=7.2)
    ax.set_xlim(0, 0.75)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="#222222")
               for c in TISSUE_COLOR.values()]
    ax.legend(handles, list(TISSUE_COLOR.keys()), fontsize=8,
              loc="lower right", title="Tissue", title_fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", lw=0.7)
    ax.set_axisbelow(True)
    save(fig, "pc_symbolic_quality")


# ── F2. Rescoring outcome per cell line ─────────────────────────────────────
def fig_rescoring_per_cellline():
    d = res.sort_values("mean_delta", ascending=True)
    colors = [TISSUE_COLOR.get(t, "#888888") for t in d.tissue]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    y = np.arange(len(d))

    a1.barh(y, d.mean_delta, color=colors, edgecolor="#222222", linewidth=0.6)
    a1.set_yticks(y); a1.set_yticklabels(d.cell_line, fontsize=8.5)
    a1.set_xlabel("Mean $\\Delta S_{\\mathrm{rescored}}$ (Loewe units)", fontweight="bold")
    a1.set_title("Mean synergy improvement", fontweight="bold", fontsize=10.5)
    for i, v in enumerate(d.mean_delta):
        a1.text(v + 0.2, i, "%.1f" % v, va="center", fontsize=7.2)
    a1.set_xlim(0, d.mean_delta.max() * 1.18)
    a1.spines[["top", "right"]].set_visible(False)
    a1.grid(axis="x", color="#EEEEEE", lw=0.7); a1.set_axisbelow(True)

    a2.barh(y, d.pct_improved, color=colors, edgecolor="#222222",
            linewidth=0.6, alpha=0.85)
    a2.set_xlabel("Candidates with $\\Delta S > 0$ (%)", fontweight="bold")
    a2.set_title("Fraction of successful replacements", fontweight="bold", fontsize=10.5)
    a2.axvline(50, color="#444444", ls="--", lw=1.0)
    for i, v in enumerate(d.pct_improved):
        a2.text(v + 0.8, i, "%.0f%%" % v, va="center", fontsize=7.2)
    a2.set_xlim(0, 108)
    a2.spines[["top", "right"]].set_visible(False)
    a2.grid(axis="x", color="#EEEEEE", lw=0.7); a2.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="#222222")
               for c in TISSUE_COLOR.values()]
    a2.legend(handles, list(TISSUE_COLOR.keys()), fontsize=8,
              loc="lower right", title="Tissue", title_fontsize=8.5)
    save(fig, "pc_rescoring_per_cellline")


# ── F3. Tissue-level summary ────────────────────────────────────────────────
def fig_tissue_summary():
    d = tis.sort_values("mean_delta", ascending=False)
    colors = [TISSUE_COLOR.get(t, "#888888") for t in d.tissue]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    x = np.arange(len(d))
    panels = [
        ("mean_delta",     "Mean $\\Delta S$ (Loewe)",        "Synergy improvement"),
        ("mean_test_pcc",  "Mean test PCC",                    "Surrogate model quality"),
        ("mean_pct_improved", "Mean % improved",               "Replacement success rate"),
        ("total_high",     "MoA-high repurposing hits",        "High-confidence candidates"),
    ]
    ticklabels = ["%s\n(n=%d)" % (t.replace("_", "\n"), n)
                  for t, n in zip(d.tissue, d.n_cell_lines)]
    for ax, (col, ylab, title) in zip(axes, panels):
        ax.bar(x, d[col], color=colors, edgecolor="#222222", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(ticklabels, fontsize=7.5, rotation=0)
        ax.set_ylabel(ylab, fontweight="bold", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=9.5)
        for i, v in enumerate(d[col]):
            ax.text(i, v, ("%.0f" if col == "total_high" else "%.2f") % v,
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_ylim(0, d[col].max() * 1.20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
    fig.suptitle("Tissue-level aggregation of pipeline outcomes",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "pc_tissue_summary")


# ── F4. Pooled before/after synergy ─────────────────────────────────────────
def fig_before_after_pooled():
    o = pd.to_numeric(pooled.teacher_original_synergy, errors="coerce")
    n = pd.to_numeric(pooled.replaced_predicted_synergy, errors="coerce")
    m = o.notna() & n.notna()
    o, n = o[m], n[m]
    tissues = pooled.loc[m, "cell_line"].map(
        dict(zip(master.cell_line, master.tissue)))
    colors = [TISSUE_COLOR.get(t, "#888888") for t in tissues]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6))

    a1.scatter(o, n, c=colors, s=11, alpha=0.55, edgecolors="none")
    lim = [min(o.min(), n.min()) - 5, max(o.max(), n.max()) + 5]
    a1.plot(lim, lim, color="#333333", ls="--", lw=1.2, label="no change")
    a1.axhline(0, color="#888888", lw=0.9)
    a1.axvline(0, color="#888888", lw=0.9)
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("Original predicted synergy (Loewe)", fontweight="bold")
    a1.set_ylabel("Post-replacement synergy (Loewe)", fontweight="bold")
    a1.set_title("Before vs. after fragment replacement\n"
                 "(points above dashed line = improvement)",
                 fontweight="bold", fontsize=10.5)
    a1.legend(fontsize=8.5, loc="upper left")
    a1.spines[["top", "right"]].set_visible(False)
    a1.grid(color="#F0F0F0", lw=0.7); a1.set_axisbelow(True)
    flips = int(((o < 0) & (n > 0)).sum())
    a1.text(0.97, 0.04,
            "n = %d candidates\n%.1f%% improved\n%d antagonism→synergy flips"
            % (len(o), 100 * (n > o).sum() / len(o), flips),
            transform=a1.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec="#999999"))

    delta = n - o
    a2.hist(delta, bins=70, color="#4878A8", edgecolor="#20344A", linewidth=0.4)
    a2.axvline(0, color="#222222", ls="--", lw=1.4)
    a2.axvline(delta.mean(), color="#D62728", lw=1.8,
               label="mean = %+.1f" % delta.mean())
    a2.set_xlabel("$\\Delta S_{\\mathrm{rescored}}$ (Loewe units)", fontweight="bold")
    a2.set_ylabel("Number of candidates", fontweight="bold")
    a2.set_title("Distribution of synergy improvement\n(all cell lines pooled)",
                 fontweight="bold", fontsize=10.5)
    a2.legend(fontsize=9)
    a2.spines[["top", "right"]].set_visible(False)
    a2.grid(axis="y", color="#F0F0F0", lw=0.7); a2.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "pc_before_after_pooled")


# ── F5. MoA funnel + high-confidence pairings ───────────────────────────────
def fig_moa_funnel():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                 gridspec_kw={"width_ratios": [1, 1.5]})

    stages = ["Reconstructed\ncandidates", "Passing\n$\\Delta S>5$",
              "Existing-drug\nfragment match", "MoA\nmedium", "MoA\nhigh"]
    vals = [3903, int((pd.to_numeric(pooled.rescored_delta_synergy_vs_model,
                                     errors="coerce") > 5).sum()),
            2433, 36, 63]
    cols = ["#B0C4DE", "#7FA6CE", "#4878A8", "#2A7A4B", "#1A5C33"]
    bars = a1.bar(range(len(stages)), vals, color=cols, edgecolor="#1A1A1A",
                  linewidth=0.9)
    a1.set_xticks(range(len(stages)))
    a1.set_xticklabels(stages, fontsize=8.2)
    a1.set_ylabel("Count (log scale)", fontweight="bold")
    a1.set_yscale("log")
    a1.set_title("Repurposing filter funnel", fontweight="bold", fontsize=10.5)
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, v * 1.12, "%d" % v,
                ha="center", fontsize=9, fontweight="bold")
    a1.set_ylim(10, 9000)
    a1.spines[["top", "right"]].set_visible(False)
    a1.grid(axis="y", color="#F0F0F0", lw=0.7); a1.set_axisbelow(True)

    d = hi.sort_values("n_cell_lines")
    labels = ["%s\n$\\rightarrow$ %s" % (r.replaced_drug[:26],
                                          r.repurpose_drug_name[:26])
              for r in d.itertuples()]
    y = np.arange(len(d))
    moa_col = {"DNA topoisomerase I inhibitor": "#C0601A",
               "Serine/threonine-protein kinase mTOR inhibitor": "#2E6DA4",
               "PI3-kinase class I inhibitor": "#5B2C8D"}
    colors = [moa_col.get(m, "#888888") for m in d.moa]
    a2.barh(y, d.n_cell_lines, color=colors, edgecolor="#1A1A1A", linewidth=0.8)
    a2.set_yticks(y); a2.set_yticklabels(labels, fontsize=8)
    a2.set_xlabel("Number of cell lines where recovered", fontweight="bold")
    a2.set_title("MoA-high repurposing pairings\n"
                 "(shared ChEMBL target with replaced drug)",
                 fontweight="bold", fontsize=10.5)
    for i, r in enumerate(d.itertuples()):
        a2.text(r.n_cell_lines + 0.25, i,
                "%d lines · mean $\\Delta S$=%+.1f" % (r.n_cell_lines, r.mean_delta),
                va="center", fontsize=7.8)
    a2.set_xlim(0, d.n_cell_lines.max() * 1.65)
    a2.spines[["top", "right"]].set_visible(False)
    a2.grid(axis="x", color="#F0F0F0", lw=0.7); a2.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="#1A1A1A")
               for c in moa_col.values()]
    a2.legend(handles, ["Topoisomerase I", "mTOR", "PI3K"], fontsize=8,
              loc="lower right", title="Shared target", title_fontsize=8.5)
    fig.tight_layout()
    save(fig, "pc_moa_funnel")


# ── F6. Rescue vs amplify ───────────────────────────────────────────────────
def fig_rescue_vs_amplify():
    d = pooled.copy()
    d["delta"] = pd.to_numeric(d.rescored_delta_synergy_vs_model, errors="coerce")
    d["orig"] = pd.to_numeric(d.teacher_original_synergy, errors="coerce")
    d = d.dropna(subset=["delta", "orig", "selection"])

    # Triangle layout: the success-rate comparison, which carries the claim,
    # sits alone on top; the magnitude and dose-response panels sit below.
    fig = plt.figure(figsize=(13.5, 10.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.34, wspace=0.26)
    a2 = fig.add_subplot(gs[0, :])
    a1 = fig.add_subplot(gs[1, 0])
    a3 = fig.add_subplot(gs[1, 1])
    palette = {"rescue": "#2A7A4B", "amplify": "#C0601A"}

    groups = ["rescue", "amplify"]
    pct = [100 * (d[d.selection == g]["delta"] > 0).mean() for g in groups]
    mean_d = [d[d.selection == g]["delta"].mean() for g in groups]
    ns = [int((d.selection == g).sum()) for g in groups]

    # top: success rate, with the denominator on every bar
    bars = a2.bar(range(2), pct, color=[palette[g] for g in groups],
                  edgecolor="#1A1A1A", linewidth=1.0, width=0.5)
    a2.set_xticks(range(2))
    a2.set_xticklabels(["rescue: antagonistic start\n(n = %s candidates)" % f"{ns[0]:,}",
                        "amplify: synergistic start\n(n = %s candidates)" % f"{ns[1]:,}"],
                       fontsize=14)
    a2.axhline(50, color="#333333", ls="--", lw=1.4)
    a2.text(-0.46, 52.0, "equal numbers improve and worsen", fontsize=11,
            color="#333333", ha="left")
    a2.set_ylabel("Candidates improved (%)", fontweight="bold", fontsize=15)
    for b, v, m, n in zip(bars, pct, mean_d, ns):
        a2.text(b.get_x() + b.get_width() / 2, v + 1.8,
                "%.1f%%  of %s\nmean $\\Delta S$ = %+.2f" % (v, f"{n:,}", m),
                ha="center", fontweight="bold", fontsize=14)
    a2.set_ylim(0, 100)
    a2.tick_params(labelsize=13)
    a2.spines[["top", "right"]].set_visible(False)
    a2.grid(axis="y", color="#F0F0F0", lw=0.7); a2.set_axisbelow(True)

    # bottom left: full distribution of the improvement, not just its sign
    data = [d[d.selection == g]["delta"].values for g in groups]
    bp = a1.boxplot(data,
                    labels=["rescue\n(n = %s)" % f"{ns[0]:,}",
                            "amplify\n(n = %s)" % f"{ns[1]:,}"],
                    patch_artist=True, widths=0.55, showfliers=False)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(palette[g]); patch.set_alpha(0.65)
        patch.set_edgecolor("#1A1A1A")
    for med in bp["medians"]:
        med.set_color("#000000"); med.set_linewidth(2.0)
    a1.axhline(0, color="#333333", ls="--", lw=1.3)
    a1.set_ylabel("$\\Delta S_{\\mathrm{rescored}}$ (Loewe)",
                  fontweight="bold", fontsize=15)
    a1.set_title("Size of the improvement", fontweight="bold", fontsize=15)
    a1.tick_params(labelsize=13)
    a1.spines[["top", "right"]].set_visible(False)
    a1.grid(axis="y", color="#F0F0F0", lw=0.7); a1.set_axisbelow(True)

    # bottom right: improvement against starting synergy
    for g in groups:
        s = d[d.selection == g]
        a3.scatter(s.orig, s.delta, s=16, alpha=0.45, label=g,
                   color=palette[g], edgecolors="none")
    a3.axhline(0, color="#333333", ls="--", lw=1.2)
    a3.axvline(0, color="#333333", ls=":", lw=1.2)
    a3.set_xlabel("Original synergy (Loewe)", fontweight="bold", fontsize=15)
    a3.set_ylabel("$\\Delta S_{\\mathrm{rescored}}$", fontweight="bold", fontsize=15)
    a3.set_title("Improvement vs. starting synergy",
                 fontweight="bold", fontsize=15)
    a3.legend(fontsize=13, loc="upper right")
    a3.tick_params(labelsize=13)
    a3.spines[["top", "right"]].set_visible(False)
    a3.grid(color="#F5F5F5", lw=0.7); a3.set_axisbelow(True)

    save(fig, "pc_rescue_vs_amplify")
    print("     rescue n=%d (%.1f%%, mean %+.2f) | amplify n=%d (%.1f%%, mean %+.2f)"
          % (ns[0], pct[0], mean_d[0], ns[1], pct[1], mean_d[1]))


# ── F7. Cross-cell-line fragment swaps ──────────────────────────────────────
def fig_cross_fragments():
    d = xfrag.head(12).sort_values("mean_delta")
    labels = ["%s\n$\\rightarrow$ %s" % (r.source_fragment_smiles,
                                          r.candidate_fragment_smiles)
              for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    y = np.arange(len(d))
    norm = plt.Normalize(d.n_cell_lines.min(), d.n_cell_lines.max())
    colors = plt.cm.viridis(norm(d.n_cell_lines))
    ax.barh(y, d.mean_delta, color=colors, edgecolor="#1A1A1A", linewidth=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.8, family="monospace")
    ax.set_xlabel("Mean $\\Delta S_{\\mathrm{rescored}}$ (Loewe units)", fontweight="bold")
    ax.set_title("Most transferable fragment substitutions\n"
                 "(recovered independently in many cell lines)",
                 fontweight="bold", fontsize=11)
    for i, r in enumerate(d.itertuples()):
        ax.text(r.mean_delta + 0.4, i, "%d lines" % r.n_cell_lines,
                va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, d.mean_delta.max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#F0F0F0", lw=0.7); ax.set_axisbelow(True)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("Cell lines recovering swap", fontweight="bold", fontsize=9)
    save(fig, "pc_cross_fragments")


# ── F8. MoA-high heatmap: pairing × cell line ───────────────────────────────
def fig_moa_heatmap():
    pool = pd.read_csv(os.path.join(AGG, "repurposing_pooled_fixed.csv"))
    h = pool[pool.moa_confidence_fixed == "high"].copy()
    h["pairing"] = (h.replaced_drug.str.slice(0, 24) + " → "
                    + h.repurpose_drug_name.str.slice(0, 24))
    piv = h.pivot_table(index="pairing", columns="cell_line",
                        values="delta_synergy", aggfunc="max")
    order = piv.notna().sum(axis=1).sort_values(ascending=False).index
    piv = piv.loc[order]
    piv = piv[sorted(piv.columns)]

    fig, ax = plt.subplots(figsize=(14, 3.6))
    data = piv.values.astype(float)
    im = ax.imshow(np.ma.masked_invalid(data), cmap="YlOrRd",
                   aspect="auto", vmin=0)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title("Mechanism-validated repurposing candidates recovered per cell line\n"
                 "(colour = best $\\Delta S$; blank = not recovered)",
                 fontweight="bold", fontsize=10.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, "%.0f" % data[i, j], ha="center", va="center",
                        fontsize=6.2, color="#000000")
    ax.set_xticks(np.arange(-.5, len(piv.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(piv.index), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("$\\Delta S$ (Loewe)", fontweight="bold", fontsize=9)
    save(fig, "pc_moa_heatmap")


# ── F9. Model quality vs outcome ────────────────────────────────────────────
def fig_quality_vs_outcome():
    d = master.copy()
    colors = [TISSUE_COLOR.get(t, "#888888") for t in d.tissue]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    a1.scatter(d.test_pcc, d.mean_delta, c=colors, s=95,
               edgecolors="#1A1A1A", linewidths=0.8, zorder=3)
    for r in d.itertuples():
        a1.annotate(r.cell_line, (r.test_pcc, r.mean_delta), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    z = np.polyfit(d.test_pcc, d.mean_delta, 1)
    xs = np.linspace(d.test_pcc.min(), d.test_pcc.max(), 50)
    r = np.corrcoef(d.test_pcc, d.mean_delta)[0, 1]
    a1.plot(xs, np.polyval(z, xs), color="#333333", ls="--", lw=1.3,
            label="r = %.2f" % r)
    a1.set_xlabel("Symbolic surrogate test PCC", fontweight="bold")
    a1.set_ylabel("Mean $\\Delta S_{\\mathrm{rescored}}$", fontweight="bold")
    a1.set_title("Surrogate quality vs. realised improvement",
                 fontweight="bold", fontsize=10.5)
    a1.legend(fontsize=9)
    a1.spines[["top", "right"]].set_visible(False)
    a1.grid(color="#F5F5F5", lw=0.7); a1.set_axisbelow(True)

    a2.scatter(d.n_reconstructed, d.n_hits, c=colors, s=95,
               edgecolors="#1A1A1A", linewidths=0.8, zorder=3)
    for r2 in d.itertuples():
        a2.annotate(r2.cell_line, (r2.n_reconstructed, r2.n_hits), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    a2.set_xlabel("Reconstructed candidates", fontweight="bold")
    a2.set_ylabel("Repurposing hits", fontweight="bold")
    a2.set_title("Reconstruction yield vs. repurposing hits",
                 fontweight="bold", fontsize=10.5)
    a2.spines[["top", "right"]].set_visible(False)
    a2.grid(color="#F5F5F5", lw=0.7); a2.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="#1A1A1A")
               for c in TISSUE_COLOR.values()]
    a2.legend(handles, list(TISSUE_COLOR.keys()), fontsize=8, loc="lower right",
              title="Tissue", title_fontsize=8.5)
    fig.tight_layout()
    save(fig, "pc_quality_vs_outcome")


if __name__ == "__main__":
    print("Generating pan-cancer figures →", OUTDIR)
    fig_symbolic_quality()
    fig_rescoring_per_cellline()
    fig_tissue_summary()
    fig_before_after_pooled()
    fig_moa_funnel()
    fig_rescue_vs_amplify()
    fig_cross_fragments()
    fig_moa_heatmap()
    fig_quality_vs_outcome()
    print("Done.")
