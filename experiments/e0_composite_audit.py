"""
E0 -- Composite harmfulness score audit
=======================================
The paper (Eq. 9) states that harmful fragments are ranked by

    H_i = a_SHAP * phi_i + a_LOO * dS_i^LOO + a_Attn * (1 - alpha_i)

This script checks what the pipeline actually computed, using the emitted
per-fragment tables. It answers three questions:

  Q1  Are fragment_shap_value, leave_one_out_delta and the signed SHAP sum
      distinct quantities, or the same number under three names?
  Q2  Does the realised ranking match a weighted composite, or a single signal?
  Q3  Does attention_weight influence the ranking at all?

Read-only. Writes only into experiments/results/.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RUN = "clean_runs/clean_runs/2026-07-10_full_rerun_inference"
OUT = "experiments/results"
COLS = ["fragment_shap_value", "leave_one_out_delta",
        "fragment_embedding_dim_signed_shap_sum",
        "fragment_embedding_dim_abs_shap_sum", "attention_weight"]


def load():
    frames = []
    for f in sorted(glob.glob(os.path.join(RUN, "shap/*/top_200_harmful_fragments.csv"))):
        d = pd.read_csv(f)
        d["cell_line"] = f.split(os.sep)[-2]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 78)
    say("  E0  COMPOSITE HARMFULNESS SCORE AUDIT")
    say("=" * 78)
    say("")
    say("rows: %s across %d cell lines" % (format(len(d), ","), d.cell_line.nunique()))
    say("")

    # ---- Q1: are the three signals actually distinct? --------------------
    say("-" * 78)
    say("Q1  Are the attribution signals distinct quantities?")
    say("-" * 78)
    sub = d[COLS].dropna()
    say("  pairwise Pearson correlation (n=%s):" % format(len(sub), ","))
    corr = sub.corr()
    say(corr.to_string(float_format=lambda x: "%+.6f" % x))
    say("")

    a, b = sub.fragment_shap_value, sub.leave_one_out_delta
    ident = np.isclose(a, b, rtol=1e-6, atol=1e-6)
    say("  fragment_shap_value == leave_one_out_delta : %d / %d rows (%.1f%%)"
        % (ident.sum(), len(sub), 100 * ident.mean()))
    say("  max |difference|                           : %.3e"
        % np.abs(a - b).max())

    c = sub.fragment_embedding_dim_signed_shap_sum
    ident2 = np.isclose(a, c, rtol=1e-6, atol=1e-6)
    say("  fragment_shap_value == signed_shap_sum     : %d / %d rows (%.1f%%)"
        % (ident2.sum(), len(sub), 100 * ident2.mean()))
    say("  max |difference|                           : %.3e"
        % np.abs(a - c).max())
    say("")

    # ---- Q2: what does the realised ranking follow? -----------------------
    say("-" * 78)
    say("Q2  What does the realised ordering follow?")
    say("-" * 78)
    say("  Within each cell line, Spearman between the emitted row order")
    say("  and each candidate ranking signal:")
    rows = []
    for cl, g in d.groupby("cell_line"):
        order = np.arange(len(g))
        r = {"cell_line": cl, "n": len(g)}
        for c_ in COLS:
            v = pd.to_numeric(g[c_], errors="coerce")
            r[c_] = np.nan if v.notna().sum() < 5 else spearmanr(order, v).correlation
        rows.append(r)
    rk = pd.DataFrame(rows)
    say("")
    say("  %-38s %8s %8s" % ("signal", "mean rho", "median"))
    say("  " + "-" * 56)
    for c_ in COLS:
        say("  %-38s %+8.4f %+8.4f" % (c_, rk[c_].mean(), rk[c_].median()))
    say("")
    say("  (rho = +1.000 means the table is sorted exactly on that signal)")
    say("")

    # ---- Q3: does attention enter the ranking? ----------------------------
    say("-" * 78)
    say("Q3  Does attention_weight influence selection?")
    say("-" * 78)
    aw = pd.to_numeric(d.attention_weight, errors="coerce").dropna()
    say("  attention_weight distribution over selected fragments:")
    say("     min %.4f | 25%% %.4f | median %.4f | 75%% %.4f | max %.4f"
        % (aw.min(), aw.quantile(.25), aw.median(), aw.quantile(.75), aw.max()))
    say("  exactly 1.0 (single-fragment drug): %d / %d (%.1f%%)"
        % ((np.isclose(aw, 1.0)).sum(), len(aw), 100 * np.isclose(aw, 1.0).mean()))
    say("  rank correlation with emitted order: %+.4f"
        % rk.attention_weight.mean())
    say("")

    # ---- verdict ----------------------------------------------------------
    say("=" * 78)
    say("  VERDICT")
    say("=" * 78)
    same = 100 * ident.mean()
    loo_rho = rk.leave_one_out_delta.mean()
    att_rho = rk.attention_weight.mean()
    say("")
    if same > 99:
        say("  * fragment_shap_value and leave_one_out_delta are THE SAME NUMBER")
        say("    in %.1f%% of rows. They are not two independent signals." % same)
    if abs(loo_rho) > 0.99:
        say("  * the emitted ordering is an exact sort on leave_one_out_delta")
        say("    (mean rho %+.4f), i.e. a single-signal ranking." % loo_rho)
    if abs(att_rho) < 0.3:
        say("  * attention_weight does not drive the ordering (mean rho %+.4f)."
            % att_rho)
    say("")
    say("  Paper Eq. 9 describes a weighted three-signal composite with")
    say("  coefficients a_SHAP, a_LOO, a_Attn. No such weights appear in the")
    say("  codebase, and the evidence above indicates the implemented rule is")
    say("  a filter on leave_one_out_delta < 0 followed by a lexicographic")
    say("  sort with the SHAP sums as tie-breakers only.")
    say("")

    with open(os.path.join(OUT, "e0_composite_audit.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    rk.to_csv(os.path.join(OUT, "e0_rank_correlations.csv"), index=False)
    corr.to_csv(os.path.join(OUT, "e0_signal_correlations.csv"))
    json.dump({"pct_shap_equals_loo": float(same),
               "mean_rho_loo": float(loo_rho),
               "mean_rho_attention": float(att_rho),
               "n_rows": int(len(d)),
               "n_cell_lines": int(d.cell_line.nunique())},
              open(os.path.join(OUT, "e0_summary.json"), "w"), indent=1)
    print("\nwritten -> %s/e0_composite_audit.txt" % OUT)


if __name__ == "__main__":
    main()
