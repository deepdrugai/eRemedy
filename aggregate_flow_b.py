"""
aggregate_flow_b.py
===================
Pool the per-cell-line Flow B outputs into one pan-cancer view.

The headline number is not the raw improvement -- substituting almost anything
into a strongly antagonistic pair looks like an improvement. What matters is
how far the model's picks beat the gain an arbitrary substitution would have
delivered, measured against the experimental data rather than against another
prediction.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="clean_runs/flow_b")
    p.add_argument("--outdir", default="thesis/aggregate")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    hit_files = sorted(glob.glob(os.path.join(args.root, "*", "flowB_hits_*.csv")))
    all_files = sorted(glob.glob(os.path.join(args.root, "*", "flowB_all_scores_*.csv")))
    if not hit_files:
        raise SystemExit("No Flow B hit files under %s" % args.root)

    hits = pd.concat([pd.read_csv(f) for f in hit_files], ignore_index=True)
    print("cell lines: %d | hits: %s" % (len(hit_files), format(len(hits), ",")))

    scores = None
    if all_files:
        scores = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
        print("all scored substitutions: %s" % format(len(scores), ","))

    col = "observed_loewe_of_recommendation"

    # ---- per cell line ----------------------------------------------------
    rows = []
    for cl, g in hits.groupby("cell_line"):
        chk = g.dropna(subset=[col]) if col in g.columns else g.iloc[0:0]
        base = np.nan
        if scores is not None and col in scores.columns:
            s = scores[scores.cell_line == cl].dropna(subset=[col])
            keys = ["drug_a", "drug_b", "replaced_side"]
            if len(s) and len(chk):
                sel = chk[keys].drop_duplicates()
                b = s.merge(sel, on=keys)
                base = float((b[col] - b.observed_loewe).mean())
        realised = float((chk[col] - chk.observed_loewe).mean()) if len(chk) else np.nan
        rows.append({
            "cell_line": cl,
            "n_hits": len(g),
            "n_moa_high": int((g.moa_confidence == "high").sum()),
            "n_novel": int(g.combination_is_novel.sum()) if "combination_is_novel" in g else 0,
            "mean_predicted_gain": g.delta_synergy.mean(),
            "n_with_ground_truth": len(chk),
            "pct_realised": 100.0 * float((chk[col] > chk.observed_loewe).mean()) if len(chk) else np.nan,
            "mean_realised_gain": realised,
            "baseline_gain": base,
            "model_advantage": realised - base if np.isfinite(base) else np.nan,
        })
    per_cell = pd.DataFrame(rows).sort_values("model_advantage", ascending=False)
    per_cell.to_csv(os.path.join(args.outdir, "flow_b_per_cellline.csv"), index=False)

    print()
    print("=" * 104)
    print("  FLOW B PER CELL LINE")
    print("=" * 104)
    print(per_cell.to_string(index=False, float_format=lambda x: "%.2f" % x))

    # ---- pooled -----------------------------------------------------------
    chk = hits.dropna(subset=[col]) if col in hits.columns else hits.iloc[0:0]
    print()
    print("=" * 104)
    print("  POOLED")
    print("=" * 104)
    print("  hits                                   : %s" % format(len(hits), ","))
    print("  MoA-high hits                          : %d" % int((hits.moa_confidence == "high").sum()))
    if "combination_is_novel" in hits.columns:
        print("  novel (never-measured) combinations    : %d" % int(hits.combination_is_novel.sum()))
    if len(chk):
        gain = chk[col] - chk.observed_loewe
        print("  hits with a measured outcome           : %d" % len(chk))
        print("  improvement realised experimentally    : %.1f%%" % (100.0 * (gain > 0).mean()))
        print("  mean realised gain                     : %+.2f" % gain.mean())
        print("  mean predicted gain                    : %+.2f" % chk.delta_synergy.mean())
        print("  corr(predicted, realised)              : %+.3f"
              % np.corrcoef(chk.delta_synergy, gain)[0, 1])
    if per_cell.model_advantage.notna().any():
        print("  mean baseline gain (any substitution)  : %+.2f" % per_cell.baseline_gain.mean())
        print("  mean model advantage over baseline     : %+.2f" % per_cell.model_advantage.mean())

    # ---- most recommended candidates --------------------------------------
    print()
    print("=" * 104)
    print("  MOST FREQUENTLY RECOMMENDED CANDIDATES  (across cell lines)")
    print("=" * 104)
    top = (hits.groupby("candidate_drug")
                .agg(n_recommendations=("candidate_drug", "size"),
                     n_cell_lines=("cell_line", "nunique"),
                     n_moa_high=("moa_confidence", lambda s: int((s == "high").sum())),
                     mean_predicted_gain=("delta_synergy", "mean"),
                     mean_z=("z_vs_null", "mean"))
                .sort_values("n_recommendations", ascending=False))
    print(top.head(20).to_string(float_format=lambda x: "%.2f" % x))
    top.to_csv(os.path.join(args.outdir, "flow_b_top_candidates.csv"), index=False)

    # ---- MoA-high shortlist ----------------------------------------------
    mh = hits[hits.moa_confidence == "high"].copy()
    if len(mh):
        mh = mh.sort_values("z_vs_null", ascending=False)
        mh.to_csv(os.path.join(args.outdir, "flow_b_moa_high.csv"), index=False)
        print()
        print("=" * 104)
        print("  MoA-HIGH SHORTLIST  (target-level mechanism match, top 25 by z)")
        print("=" * 104)
        show = ["cell_line", "drug_a", "drug_b", "replaced_drug", "candidate_drug",
                "original_model_synergy", "repurposed_synergy", "delta_synergy",
                "z_vs_null", "combination_is_novel"]
        show = [c for c in show if c in mh.columns]
        print(mh[show].head(25).to_string(index=False, float_format=lambda x: "%.2f" % x))

    print()
    print("written -> %s/flow_b_*.csv" % args.outdir)


if __name__ == "__main__":
    main()
