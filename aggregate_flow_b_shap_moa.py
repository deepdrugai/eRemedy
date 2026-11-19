"""
aggregate_flow_b_shap_moa.py
============================
Pool the per-cell-line results of the SHAP-driven, mechanism-matched Flow B
and produce the tables and figures for the paper section.

Reported separately for candidates inside and outside the teacher's 21-drug
training vocabulary, because external validation on NCI-ALMANAC put accuracy at
PCC 0.47 inside it and 0.04 outside. Recommendations that happen to have a
measured outcome in O'Neil are checked against that measurement.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="clean_runs/flow_b_shap_moa")
    p.add_argument("--outdir", default="thesis/aggregate")
    p.add_argument("--figdir", default="thesis/figures")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    hit_files = sorted(glob.glob(os.path.join(args.root, "*", "flowB_shap_moa_hits_*.csv")))
    all_files = sorted(glob.glob(os.path.join(args.root, "*", "flowB_shap_moa_all_*.csv")))
    if not hit_files:
        raise SystemExit("No results under %s" % args.root)

    hits = pd.concat([pd.read_csv(f) for f in hit_files], ignore_index=True)
    allsc = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True) if all_files else None

    print("=" * 100)
    print("  FLOW B -- SHAP-DRIVEN, MECHANISM-MATCHED REPLACEMENT: PAN-CANCER SUMMARY")
    print("=" * 100)
    print("  cell lines                       : %d" % hits.cell_line.nunique())
    if allsc is not None:
        print("  mechanism-matched substitutions  : %s" % format(len(allsc), ","))
        print("  fragment/target sites            : %d"
              % allsc.groupby(["cell_line", "replaced_drug", "retained_partner",
                               "harmful_fragment"]).ngroups)
    print("  hits (delta gate, fragment removed): %s" % format(len(hits), ","))
    print("    mechanism match high           : %d" % int((hits.moa_confidence == "high").sum()))
    print("    mechanism match medium         : %d" % int((hits.moa_confidence == "medium").sum()))
    print("    inside teacher vocabulary      : %d" % int(hits.candidate_in_teacher_vocab.sum()))
    print("    outside teacher vocabulary     : %d" % int((~hits.candidate_in_teacher_vocab).sum()))

    # ---- ground truth -----------------------------------------------------
    col, ocol = "observed_loewe_recommendation", "observed_loewe_original"
    if col in hits.columns:
        chk = hits.dropna(subset=[col, ocol])
        if len(chk):
            gain = chk[col] - chk[ocol]
            print()
            print("-" * 100)
            print("  GROUND TRUTH  (recommended combination also measured in O'Neil)")
            print("-" * 100)
            print("  n                                : %d" % len(chk))
            print("  improvement realised             : %.1f%%" % (100.0 * (gain > 0).mean()))
            print("  mean realised gain               : %+.2f" % gain.mean())
            print("  mean predicted gain              : %+.2f" % chk.delta_synergy.mean())
            if len(chk) > 2:
                print("  corr(predicted, realised)        : %+.3f"
                      % np.corrcoef(chk.delta_synergy, gain)[0, 1])
            if allsc is not None and ocol in allsc.columns:
                ab = allsc.dropna(subset=[col, ocol])
                if len(ab):
                    bg = (ab[col] - ab[ocol])
                    print("  baseline: any MoA-matched swap   : %+.2f  (n=%d)" % (bg.mean(), len(bg)))
                    print("  advantage of selected hits       : %+.2f" % (gain.mean() - bg.mean()))

    # ---- per cell line ----------------------------------------------------
    per = hits.groupby("cell_line").agg(
        n_hits=("candidate_drug", "size"),
        n_high=("moa_confidence", lambda s: int((s == "high").sum())),
        n_in_vocab=("candidate_in_teacher_vocab", "sum"),
        mean_delta=("delta_synergy", "mean"),
        max_delta=("delta_synergy", "max"),
        mean_z=("z_vs_moa_pool", "mean"),
    ).sort_values("n_hits", ascending=False)
    per.to_csv(os.path.join(args.outdir, "flow_b_shap_moa_per_cellline.csv"))
    print()
    print("-" * 100)
    print("  PER CELL LINE")
    print("-" * 100)
    print(per.to_string(float_format=lambda x: "%.2f" % x))

    # ---- recurring recommendations ---------------------------------------
    rec = hits.groupby("candidate_drug").agg(
        n=("cell_line", "size"),
        n_cells=("cell_line", "nunique"),
        n_high=("moa_confidence", lambda s: int((s == "high").sum())),
        in_vocab=("candidate_in_teacher_vocab", "first"),
        mean_delta=("delta_synergy", "mean"),
        mean_z=("z_vs_moa_pool", "mean"),
    ).sort_values("n", ascending=False)
    rec.to_csv(os.path.join(args.outdir, "flow_b_shap_moa_top_candidates.csv"))
    print()
    print("-" * 100)
    print("  MOST RECOMMENDED CANDIDATES")
    print("-" * 100)
    print(rec.head(20).to_string(float_format=lambda x: "%.2f" % x))

    # ---- recurring harmful fragments -------------------------------------
    frag = hits.groupby("harmful_fragment").agg(
        n=("cell_line", "size"),
        n_cells=("cell_line", "nunique"),
        mean_shap=("fragment_shap_value", "mean"),
        mean_delta=("delta_synergy", "mean"),
    ).sort_values("n", ascending=False)
    frag.to_csv(os.path.join(args.outdir, "flow_b_shap_moa_fragments.csv"))
    print()
    print("-" * 100)
    print("  HARMFUL FRAGMENTS DRIVING THE MOST REPLACEMENTS")
    print("-" * 100)
    print(frag.head(15).to_string(float_format=lambda x: "%.2f" % x))

    hits.sort_values("z_vs_moa_pool", ascending=False).to_csv(
        os.path.join(args.outdir, "flow_b_shap_moa_all_hits.csv"), index=False)

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    iv = hits[hits.candidate_in_teacher_vocab]
    ov = hits[~hits.candidate_in_teacher_vocab]
    ax[0].hist([iv.delta_synergy, ov.delta_synergy], bins=25, stacked=True,
               color=["#1a7f37", "#b8860b"], edgecolor="black",
               label=["in teacher vocabulary", "outside vocabulary"])
    ax[0].set_xlabel("predicted synergy gain", fontsize=11, color="black")
    ax[0].set_ylabel("recommendations", fontsize=11, color="black")
    ax[0].set_title("Gain by teacher reliability", fontsize=12,
                    fontweight="bold", color="black")
    ax[0].legend(fontsize=9)

    t = per.head(15)
    ax[1].barh(range(len(t)), t.n_hits.values, color="#1a5fb4",
               edgecolor="black", linewidth=1.0)
    ax[1].set_yticks(range(len(t)))
    ax[1].set_yticklabels(t.index, fontsize=9)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("recommendations", fontsize=11, color="black")
    ax[1].set_title("Per cell line", fontsize=12, fontweight="bold", color="black")

    tr = rec.head(12)
    cols = ["#1a7f37" if v else "#b8860b" for v in tr.in_vocab]
    ax[2].barh(range(len(tr)), tr.n.values, color=cols, edgecolor="black", linewidth=1.0)
    ax[2].set_yticks(range(len(tr)))
    ax[2].set_yticklabels([str(s)[:22] for s in tr.index], fontsize=9)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("times recommended", fontsize=11, color="black")
    ax[2].set_title("Most recommended drugs", fontsize=12,
                    fontweight="bold", color="black")

    for a in ax:
        a.tick_params(colors="black", labelsize=9)
        for s in a.spines.values():
            s.set_color("black")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(args.figdir, "flow_b_shap_moa_overview." + ext),
                    dpi=200, bbox_inches="tight")
    plt.close()

    print()
    print("written -> %s/flow_b_shap_moa_*.csv" % args.outdir)
    print("figure  -> %s/flow_b_shap_moa_overview.{pdf,png}" % args.figdir)


if __name__ == "__main__":
    main()
