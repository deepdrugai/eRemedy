#!/usr/bin/env python3
"""
finalize_flow_b.py
==================
Turn the raw Flow B output into the tables a reviewer can actually read.

Three changes to how the results are presented:

  1. One recommendation per site. A "site" is a (cell line, harmful fragment,
     drug carrying it, retained partner) tuple -- the unit the pipeline
     actually acts on. Reporting every mechanism-matched pool member above a
     loose delta gate inflated the result into tens of thousands of rows that
     say little; the top-ranked candidate per site is the real deliverable.

  2. The measured subset leads. Where the recommended combination also exists
     in O'Neil, the outcome is an experimental measurement and the model is
     only doing selection. That is the only place in the pipeline where a
     synergy number carries no model uncertainty, so it is reported first.

  3. Everything is split by teacher vocabulary. External validation on
     NCI-ALMANAC put accuracy at PCC 0.47 for drugs inside the 21-drug
     training vocabulary and 0.04 outside it. In-vocabulary recommendations
     carry the quantitative claim; the rest are screening hypotheses and are
     labelled as such rather than mixed into the same numbers.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SITE_KEYS = ["cell_line", "harmful_fragment", "replaced_drug", "retained_partner"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hits", default="thesis/aggregate/flow_b_shap_moa_all_hits.csv")
    p.add_argument("--all-scores",
                   default="clean_runs/flow_b_shap_moa/*/flowB_shap_moa_all_*.csv",
                   help="glob of unfiltered per-cell-line scores, used for the "
                        "funnel counts and the arbitrary-substitution baseline")
    p.add_argument("--outdir", default="thesis/aggregate")
    p.add_argument("--figdir", default="thesis/figures")
    return p.parse_args()


def funnel_and_baseline(pattern):
    """Counts before the delta gate, and the gain an arbitrary pool member gives.

    Every mechanism-matched pool member is scored, so the within-pool mean delta
    is what a substitution chosen at random would have produced. The difference
    between that and the model's pick is the part of the gain attributable to
    the model rather than to regression to the mean.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = d.groupby(SITE_KEYS).delta_synergy.agg(["mean", "max"])
    return {"n_scored": len(d), "n_sites_attempted": len(g),
            "arbitrary": g["mean"].mean(), "selected": g["max"].mean()}


def gt_block(df, label, lines):
    """Ground-truth performance for a subset, or a note that there is none."""
    col, ocol = "observed_loewe_recommendation", "observed_loewe_original"
    chk = df.dropna(subset=[col, ocol])
    if chk.empty:
        lines.append("  %-22s  no measured outcomes" % label)
        return None
    gain = chk[col] - chk[ocol]
    r = (np.corrcoef(chk.delta_synergy, gain)[0, 1] if len(chk) > 2 else np.nan)
    lines.append("  %-22s  n=%-6d  realised %5.1f%%  gain %+6.2f  predicted %+6.2f  r=%+.3f"
                 % (label, len(chk), 100.0 * (gain > 0).mean(),
                    gain.mean(), chk.delta_synergy.mean(), r))
    return {"n": len(chk), "realised_pct": 100.0 * (gain > 0).mean(),
            "mean_gain": gain.mean(), "mean_pred": chk.delta_synergy.mean(), "r": r}


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    h = pd.read_csv(args.hits)
    print("input hits: %s" % format(len(h), ","))
    fb = funnel_and_baseline(args.all_scores)

    # ---- 1. one recommendation per site -----------------------------------
    # head(1) keeps whole rows. GroupBy.first() must not be used here: it takes
    # the first non-null value column by column, which silently splices fields
    # from different candidates into one row.
    h = h.sort_values("delta_synergy", ascending=False)
    primary = h.groupby(SITE_KEYS, as_index=False, sort=False).head(1)
    primary = primary.sort_values("z_vs_moa_pool", ascending=False)

    cols = ["cell_line", "drug_a", "drug_b", "harmful_fragment", "fragment_shap_value",
            "replaced_drug", "retained_partner", "candidate_drug", "candidate_moa",
            "moa_confidence", "candidate_in_teacher_vocab", "pool_size",
            "original_model_synergy", "repurposed_synergy", "delta_synergy",
            "z_vs_moa_pool", "observed_loewe_original", "observed_loewe_recommendation"]
    cols = [c for c in cols if c in primary.columns]
    primary[cols].to_csv(os.path.join(args.outdir, "flow_b_primary_recommendations.csv"),
                         index=False)

    iv = primary[primary.candidate_in_teacher_vocab]
    ov = primary[~primary.candidate_in_teacher_vocab]

    L = []
    L.append("=" * 96)
    L.append("  FLOW B -- FINAL RESULT TABLES")
    L.append("=" * 96)
    L.append("")
    L.append("One recommendation per site, where a site is a unique combination of cell")
    L.append("line, SHAP-identified harmful fragment, the drug carrying it, and the")
    L.append("retained partner.")
    L.append("")
    if fb:
        L.append("  substitutions scored               : %s  (across %s sites attempted)"
                 % (format(fb["n_scored"], ","), format(fb["n_sites_attempted"], ",")))
        L.append("  improving the prediction           : %s" % format(len(h), ","))
    L.append("  sites with a recommendation        : %s" % format(len(primary), ","))
    L.append("  cell lines                         : %d" % primary.cell_line.nunique())
    L.append("  distinct harmful fragments         : %d" % primary.harmful_fragment.nunique())
    L.append("  distinct recommended drugs         : %d" % primary.candidate_drug.nunique())
    L.append("  mean mechanism-matched pool size   : %.0f" % primary.pool_size.mean())
    L.append("")

    # ---- 3. vocabulary split ----------------------------------------------
    L.append("-" * 96)
    L.append("SPLIT BY TEACHER VOCABULARY")
    L.append("-" * 96)
    L.append("External validation on NCI-ALMANAC: PCC 0.47 for drug pairs inside the")
    L.append("teacher's 21-drug training vocabulary, 0.04 outside it. The in-vocabulary")
    L.append("rows carry the quantitative claim; the rest are screening hypotheses.")
    L.append("")
    L.append("  %-26s %8s %12s %10s %10s" % ("", "sites", "mean delta", "mean z", "MoA-high"))
    L.append("  " + "-" * 70)
    for lab, s in (("in teacher vocabulary", iv), ("outside vocabulary", ov)):
        if s.empty:
            continue
        L.append("  %-26s %8d %12.2f %10.2f %10d"
                 % (lab, len(s), s.delta_synergy.mean(), s.z_vs_moa_pool.mean(),
                    int((s.moa_confidence == "high").sum())))
    L.append("")

    # ---- 2. measured subset leads -----------------------------------------
    L.append("-" * 96)
    L.append("GROUND TRUTH -- RECOMMENDATIONS WHOSE OUTCOME WAS MEASURED")
    L.append("-" * 96)
    L.append("For these the model only chose the candidate; the synergy value is an")
    L.append("O'Neil measurement, so no model uncertainty enters the outcome.")
    L.append("")
    summ = {}
    summ["all_hits"] = gt_block(h, "all hits", L)
    summ["primary"] = gt_block(primary, "primary (1 per site)", L)
    summ["primary_in_vocab"] = gt_block(iv, "  in vocabulary", L)
    summ["primary_out_vocab"] = gt_block(ov, "  outside vocabulary", L)
    L.append("")

    col, ocol = "observed_loewe_recommendation", "observed_loewe_original"
    measured = primary.dropna(subset=[col, ocol]).copy()
    if not measured.empty:
        measured["realised_gain"] = measured[col] - measured[ocol]
        measured = measured.sort_values("realised_gain", ascending=False)
        measured[cols + ["realised_gain"]].to_csv(
            os.path.join(args.outdir, "flow_b_measured_recommendations.csv"), index=False)
        L.append("  written: flow_b_measured_recommendations.csv (%d rows)" % len(measured))
        L.append("")

    # ---- headline tables --------------------------------------------------
    def table(df, title, n=15):
        L.append("-" * 96)
        L.append(title)
        L.append("-" * 96)
        if df.empty:
            L.append("  (none)")
            L.append("")
            return
        for i, r in enumerate(df.head(n).itertuples(index=False), 1):
            tag = "in-vocab" if r.candidate_in_teacher_vocab else "hypothesis"
            L.append("%3d. %-10s %s + %s" % (i, r.cell_line, r.drug_a, r.drug_b))
            L.append("     fragment %s (SHAP %.1f) on %s"
                     % (str(r.harmful_fragment)[:40], r.fragment_shap_value, r.replaced_drug))
            L.append("     -> %s [%s, %s]  %+.1f -> %+.1f  (delta %+.1f, z %.2f)"
                     % (r.candidate_drug, r.moa_confidence, tag,
                        r.original_model_synergy, r.repurposed_synergy,
                        r.delta_synergy, r.z_vs_moa_pool))
            gain = getattr(r, "realised_gain", None)
            if gain is not None and np.isfinite(gain):
                L.append("     measured outcome: %+.1f -> %+.1f  (realised %+.1f)"
                         % (r.observed_loewe_original, r.observed_loewe_recommendation, gain))
            L.append("")

    if not measured.empty:
        table(measured, "TABLE 1  MEASUREMENT-BACKED RECOMMENDATIONS (ranked by realised gain)")
    table(iv, "TABLE 2  IN-VOCABULARY RECOMMENDATIONS (ranked by z)")
    table(ov[ov.moa_confidence == "high"],
          "TABLE 3  SCREENING HYPOTHESES, shared-target mechanism match (ranked by z)")

    # recurring fragments among primary sites
    L.append("-" * 96)
    L.append("HARMFUL FRAGMENTS BY NUMBER OF SITES")
    L.append("-" * 96)
    fr = primary.groupby("harmful_fragment").agg(
        sites=("cell_line", "size"), cells=("cell_line", "nunique"),
        mean_shap=("fragment_shap_value", "mean"),
        mean_delta=("delta_synergy", "mean")).sort_values("sites", ascending=False)
    for f, r in fr.head(12).iterrows():
        L.append("  %-52s sites %3d  cells %2d  SHAP %+7.1f  delta %+6.1f"
                 % (str(f)[:52], r.sites, r.cells, r.mean_shap, r.mean_delta))
    fr.to_csv(os.path.join(args.outdir, "flow_b_primary_fragments.csv"))
    L.append("")

    # ---- caveats -----------------------------------------------------------
    # The headline numbers above read far stronger than they are without these.
    L.append("-" * 96)
    L.append("HOW TO READ THESE NUMBERS")
    L.append("-" * 96)
    if fb:
        share = 100.0 * fb["arbitrary"] / fb["selected"]
        L.append("1. Regression to the mean accounts for most of the gain. Every pool")
        L.append("   member is scored, so the within-pool mean is what an arbitrary")
        L.append("   mechanism-matched substitution would have given:")
        L.append("")
        L.append("     arbitrary substitution (pool mean)  %+6.2f   %3.0f%% of selected"
                 % (fb["arbitrary"], share))
        L.append("     model-selected candidate            %+6.2f" % fb["selected"])
        L.append("     attributable to the model           %+6.2f   %3.0f%% of selected"
                 % (fb["selected"] - fb["arbitrary"], 100.0 - share))
        L.append("")
        L.append("   Starting from a strongly antagonistic pair, almost any change")
        L.append("   raises the score. Use the z column, not the raw delta, to compare")
        L.append("   candidates: z is measured against that pool null.")
        L.append("")
    L.append("2. \"100% realised\" is a selection result, not an accuracy result. For")
    L.append("   the measured rows the model only picked the candidate; it did not")
    L.append("   predict the value. It says the ranking was right every time on that")
    L.append("   subset, not that the predicted magnitudes were right.")
    L.append("")
    L.append("3. The measured rows are necessarily in-vocabulary, since only O'Neil")
    L.append("   drugs have O'Neil measurements. They are not a prospective test:")
    L.append("   the teacher had seen those combinations in training. The outcome is")
    L.append("   a measurement and so is not circular, but a genuinely prospective")
    L.append("   validation is still outstanding.")
    L.append("")
    L.append("4. The %s out-of-vocabulary rows carry no quantitative weight. On"
             % format(len(ov), ","))
    L.append("   NCI-ALMANAC the teacher ranks unseen drugs at chance. Treat them as")
    L.append("   a screening shortlist only.")
    L.append("")

    out = os.path.join(args.outdir, "flow_b_final_report.txt")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:60]))
    print("...")
    print("full report -> %s" % out)

    # ---- figure -----------------------------------------------------------
    # The measurement-backed panel carries the claim, so it gets the area; the
    # two supporting bar panels are stacked beside it.
    fig = plt.figure(figsize=(15.0, 8.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0],
                          hspace=0.42, wspace=0.24)
    ax = [fig.add_subplot(gs[0, 1]),   # deliverable split, upper right
          fig.add_subplot(gs[:, 0]),   # predicted vs measured, large, left
          fig.add_subplot(gs[1, 1])]   # fragment sites, lower right

    ax[0].bar(["in\nvocabulary", "screening\nhypotheses"], [len(iv), len(ov)],
              color=["#1a7f37", "#b8860b"], edgecolor="black", linewidth=1.2)
    # Counts sit inside the bars; the taller one would otherwise run into the
    # top of the axes.
    top = max(len(iv), len(ov))
    for i, v in enumerate([len(iv), len(ov)]):
        inside = v > 0.15 * top
        ax[0].text(i, v - 0.06 * top if inside else v + 0.03 * top, str(v),
                   ha="center", va="top" if inside else "bottom",
                   fontsize=14, fontweight="bold",
                   color="white" if inside else "black")
    ax[0].set_ylim(0, top * 1.08)
    ax[0].set_ylabel("recommendations\n(one per site)", fontsize=13, color="black")
    ax[0].set_title("Deliverable by teacher reliability", fontsize=14,
                    fontweight="bold", color="black")

    if not measured.empty:
        ax[1].scatter(measured.delta_synergy, measured.realised_gain, s=46,
                      c=["#1a7f37" if v else "#b8860b"
                         for v in measured.candidate_in_teacher_vocab],
                      edgecolor="black", linewidth=0.5, alpha=0.8)
        lim = [min(measured.delta_synergy.min(), measured.realised_gain.min()),
               max(measured.delta_synergy.max(), measured.realised_gain.max())]
        ax[1].plot(lim, lim, "k--", lw=1)
        rr = np.corrcoef(measured.delta_synergy, measured.realised_gain)[0, 1]
        ax[1].set_xlabel("predicted gain (Loewe)", fontsize=14, color="black")
        ax[1].set_ylabel("measured gain (Loewe)", fontsize=14, color="black")
        ax[1].set_title("Predicted vs measured (r = %+.2f, n = %d)" % (rr, len(measured)),
                        fontsize=15, fontweight="bold", color="black")

    # Label fragments F1..F10 rather than by SMILES: the strings are long
    # enough to collide with the neighbouring panel. The key is printed below
    # and belongs in the caption.
    t = fr.head(10)
    keys = ["F%d" % (i + 1) for i in range(len(t))]
    ax[2].barh(range(len(t)), t.sites.values, color="#1a5fb4",
               edgecolor="black", linewidth=1.0)
    ax[2].set_yticks(range(len(t)))
    ax[2].set_yticklabels(keys, fontsize=13)
    ax[2].invert_yaxis()
    for i, v in enumerate(t.sites.values):
        ax[2].text(v + max(t.sites.values) * 0.015, i, str(int(v)),
                   va="center", fontsize=11, color="black")
    ax[2].set_xlim(0, max(t.sites.values) * 1.12)
    ax[2].set_xlabel("replacement sites", fontsize=13, color="black")
    ax[2].set_title("Harmful fragments driving sites", fontsize=14,
                    fontweight="bold", color="black")

    print("\nfragment key for the Route B figure caption:")
    for k, (smi, row) in zip(keys, t.iterrows()):
        print("  %-4s %-55s %d sites" % (k, smi, int(row.sites)))

    for a in ax:
        a.tick_params(colors="black", labelsize=12)
        for s in a.spines.values():
            s.set_color("black")
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(args.figdir, "flow_b_final." + ext),
                    dpi=200, bbox_inches="tight")
    plt.close()
    print("figure -> %s/flow_b_final.{pdf,png}" % args.figdir)


if __name__ == "__main__":
    main()
