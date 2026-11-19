"""
Distribution of Loewe scores across the 23,872 retained O'Neil samples.
=======================================================================
Main-text figure: single-column histogram with the additive expectation marked
and the at-or-below-additive region shaded, which is the region both flows set
out to repair.

Supplementary figure: coverage of the 21 x 34 drug-by-cell-line grid, supporting
the sparsity claim in the Introduction.

Visual style follows gen_figures_pancancer.py (the pc_* figures).

Run:  python thesis/gen_fig_loewe_distribution.py
Outputs: thesis/figures/pc_loewe_distribution.{svg,pdf,png}
         thesis/figures/pc_grid_coverage.{svg,pdf,png}
"""

import importlib.util
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTDIR = HERE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
RUN0 = REPO / "clean_runs/2026-04-29_full_rerun"
DATASET = "/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl"

plt.rcParams.update({
    "font.size": 10,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#333333",
    "savefig.bbox": "tight",
})

ANTAG = "#D62728"
SYN = "#1F77B4"


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUTDIR / ("%s.%s" % (name, ext)), dpi=200)
    plt.close(fig)
    print("  wrote", name)


def load_retained():
    """The exact 23,872 rows the teacher trains on."""
    spec = importlib.util.spec_from_file_location("sp", REPO / "synergy_prediction.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    ex = sp._build_examples(
        dataset=sp.load_pickle(Path(DATASET)),
        molecule_embeddings=sp.load_embedding_pickle(
            RUN0 / "molecule_ae/molecule_smiles_embeddings.pkl"),
        fragment_embeddings=sp.load_embedding_pickle(
            RUN0 / "fragment_ae/fragment_smiles_embeddings.pkl"),
        fragment_map=sp.load_pickle(RUN0 / "fragment_ae/smiles_to_fragments.pkl"),
        drug_smiles_map=sp.load_drug_smiles_map(REPO / "everything_drug_smiles_map.csv"),
        include_no_fragments=False,
    )
    return ex["meta"].reset_index(drop=True), np.asarray(ex["targets"], dtype=float)


def fig_distribution(y, stats):
    fig, ax = plt.subplots(figsize=(3.5, 2.75))

    lo, hi = y.min(), y.max()
    bins = np.linspace(lo, hi, 61)
    counts, edges = np.histogram(y, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    colors = [ANTAG if c <= 0 else SYN for c in centres]
    ax.bar(centres, counts, width=np.diff(edges), color=colors,
           edgecolor="#222222", linewidth=0.25, align="center")

    ax.axvspan(lo - 2, 0, color=ANTAG, alpha=0.07, zorder=0)
    ax.axvline(0, color="#222222", ls="--", lw=1.2, zorder=5)

    ymax = counts.max()
    ax.annotate("additive\nexpectation", xy=(0, ymax * 0.97),
                xytext=(10, ymax * 0.90), fontsize=7.2, va="top", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#222222"))
    ax.text(lo + 3, ymax * 0.62,
            "%.1f%% at or below\nadditive" % stats["pct_le_zero"],
            fontsize=7.6, color=ANTAG, fontweight="bold", va="top")

    ax.set_xlabel("Loewe synergy score", fontweight="bold", fontsize=10)
    ax.set_ylabel("Number of samples", fontweight="bold", fontsize=10)
    ax.set_xlim(lo - 2, hi + 2)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", lw=0.7)
    ax.set_axisbelow(True)
    save(fig, "pc_loewe_distribution")


def fig_grid_coverage(meta):
    a = meta.drug_a.astype(str).str.strip()
    b = meta.drug_b.astype(str).str.strip()
    cl = meta.cell_line_name.astype(str).str.strip()
    drugs = sorted(set(a) | set(b))
    cells = sorted(set(cl))
    M = pd.DataFrame(0, index=drugs, columns=cells, dtype=int)
    for d, c in zip(a, cl):
        M.at[d, c] += 1
    for d, c in zip(b, cl):
        M.at[d, c] += 1

    order = M.sum(1).sort_values(ascending=False).index
    M = M.loc[order]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(M.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.5)
    ax.set_xlabel("Cell line", fontweight="bold", fontsize=9)
    ax.set_ylabel("Drug", fontweight="bold", fontsize=9)
    ax.set_title("Coverage of the 21 $\\times$ 34 drug-by-cell-line grid "
                 "(combinations per cell)", fontweight="bold", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("samples", fontsize=8)
    save(fig, "pc_grid_coverage")
    return M


def main():
    meta, y = load_retained()
    n = len(y)

    stats = {
        "n": int(n),
        "n_drugs": int(len(set(meta.drug_a) | set(meta.drug_b))),
        "n_cells": int(meta.cell_line_name.nunique()),
        "pct_le_zero": float(100 * (y <= 0).mean()),
        "n_le_zero": int((y <= 0).sum()),
        "p5": float(np.percentile(y, 5)),
        "p95": float(np.percentile(y, 95)),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "median": float(np.median(y)),
        "sd": float(y.std(ddof=1)),
        "skew": float(pd.Series(y).skew()),
    }

    fig_distribution(y, stats)
    M = fig_grid_coverage(meta)

    pairs = {tuple(sorted(t)) for t in zip(meta.drug_a.astype(str).str.strip(),
                                           meta.drug_b.astype(str).str.strip())}
    d = stats["n_drugs"]
    stats["pairs_observed"] = int(len(pairs))
    stats["pairs_possible"] = int(d * (d - 1) / 2)
    stats["pair_coverage_pct"] = 100.0 * len(pairs) / (d * (d - 1) / 2)
    stats["grid_cells_empty"] = int((M.values == 0).sum())
    stats["grid_cells_total"] = int(M.size)

    json.dump(stats, open(OUTDIR / "pc_loewe_distribution_stats.json", "w"), indent=1)

    print("\n" + "=" * 58)
    print("  NUMBERS FOR THE PROSE")
    print("=" * 58)
    print("  samples                : %s" % format(stats["n"], ","))
    print("  at or below additive   : %.1f%%  (%s samples)"
          % (stats["pct_le_zero"], format(stats["n_le_zero"], ",")))
    print("  5th / 95th percentile  : %.1f  /  %+.1f" % (stats["p5"], stats["p95"]))
    print("  minimum / maximum      : %.1f  /  %+.1f" % (stats["min"], stats["max"]))
    print("  mean / median / SD     : %.2f / %.2f / %.2f"
          % (stats["mean"], stats["median"], stats["sd"]))
    print("  skew                   : %.2f" % stats["skew"])
    print("  drug pairs observed    : %d of %d possible (%.0f%%)"
          % (stats["pairs_observed"], stats["pairs_possible"],
             stats["pair_coverage_pct"]))
    print("  empty grid cells       : %d of %d"
          % (stats["grid_cells_empty"], stats["grid_cells_total"]))
    print("=" * 58)


if __name__ == "__main__":
    main()
