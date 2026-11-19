"""
E2 -- Is the 96-case subset biased, and does the bias compound with RTM?
========================================================================
The 96 measurement-backed recommendations report a mean measured gain of
+50.99 against +38.45 for the full set of 1,286. If the 96 also start more
antagonistic than the rest, then regression to the mean explains a larger
share of their improvement than the pooled 63% figure implies, and the two
biases stack rather than being alternatives.

Read-only. Writes only into experiments/results/.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "experiments/results"
PRIMARY = "thesis/aggregate/flow_b_primary_recommendations.csv"
MEASURED = "thesis/aggregate/flow_b_measured_recommendations.csv"
POOLS = "clean_runs/flow_b_shap_moa/*/flowB_shap_moa_all_*.csv"
SITE = ["cell_line", "harmful_fragment", "replaced_drug", "retained_partner"]


def main():
    os.makedirs(OUT, exist_ok=True)
    p = pd.read_csv(PRIMARY)
    mm = pd.read_csv(MEASURED)
    key = set(map(tuple, mm[SITE].astype(str).values))
    p["measured"] = [tuple(r) in key for r in p[SITE].astype(str).values]

    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 78)
    say("  E2  IS THE 96-CASE SUBSET BIASED?")
    say("=" * 78)
    say("")

    g_m = p[p.measured]
    g_o = p[~p.measured]
    say("-" * 78)
    say("A  Baseline antagonism: measured subset vs the rest")
    say("-" * 78)
    say("  %-26s %8s %8s %8s" % ("", "measured", "other", "diff"))
    say("  " + "-" * 54)
    for col, lab in [("original_model_synergy", "baseline synergy"),
                     ("delta_synergy", "predicted gain"),
                     ("z_vs_moa_pool", "within-pool z"),
                     ("pool_size", "pool size")]:
        if col not in p.columns:
            continue
        a, b = g_m[col].mean(), g_o[col].mean()
        say("  %-26s %8.2f %8.2f %+8.2f" % (lab, a, b, a - b))
    say("  %-26s %8d %8d" % ("n", len(g_m), len(g_o)))
    say("")

    t = stats.mannwhitneyu(g_m.original_model_synergy.dropna(),
                           g_o.original_model_synergy.dropna(),
                           alternative="two-sided")
    say("  Mann-Whitney on baseline synergy: U=%.0f, p=%.3g" % (t.statistic, t.pvalue))
    d = ((g_m.original_model_synergy.mean() - g_o.original_model_synergy.mean())
         / np.sqrt((g_m.original_model_synergy.var() + g_o.original_model_synergy.var()) / 2))
    say("  Cohen's d: %+.3f" % d)
    say("")

    # ---- RTM decomposition restricted to the measured sites ---------------
    say("-" * 78)
    say("B  Does regression to the mean explain more for the 96?")
    say("-" * 78)
    frames = [pd.read_csv(f) for f in sorted(glob.glob(POOLS))]
    allsc = pd.concat(frames, ignore_index=True)
    pool = (allsc.groupby(SITE).delta_synergy
            .agg(pool_mean="mean", pool_max="max", n="count").reset_index())
    pool["measured"] = [tuple(r) in key for r in pool[SITE].astype(str).values]

    say("  %-22s %10s %10s %10s %8s" %
        ("stratum", "arbitrary", "selected", "model", "share"))
    say("  " + "-" * 64)
    rows = []
    for lab, sub in [("measured (96)", pool[pool.measured]),
                     ("all other sites", pool[~pool.measured]),
                     ("ALL sites", pool)]:
        arb, sel = sub.pool_mean.mean(), sub.pool_max.mean()
        say("  %-22s %+10.2f %+10.2f %+10.2f %7.0f%%"
            % (lab, arb, sel, sel - arb, 100 * arb / sel))
        rows.append(dict(stratum=lab, n=len(sub), arbitrary=arb, selected=sel,
                         model=sel - arb, rtm_share=100 * arb / sel))
    say("")
    say("  'share' is the fraction of the selected gain reproducible by an")
    say("  arbitrary mechanism-matched substitution.")
    say("")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e2_rtm_by_stratum.csv"), index=False)

    # ---- realised vs predicted on the measured subset ---------------------
    say("-" * 78)
    say("C  Measured subset: realised gain against its own baseline")
    say("-" * 78)
    mm2 = mm.copy()
    r_base = stats.pearsonr(mm2.observed_loewe_original, mm2.realised_gain)
    say("  baseline (measured) : mean %+.2f | range %+.1f .. %+.1f"
        % (mm2.observed_loewe_original.mean(),
           mm2.observed_loewe_original.min(), mm2.observed_loewe_original.max()))
    say("  realised gain       : mean %+.2f" % mm2.realised_gain.mean())
    say("  corr(baseline, realised gain): r=%+.3f  p=%.3g"
        % (r_base[0], r_base[1]))
    say("")
    say("  A strongly negative correlation means the largest realised gains")
    say("  come from the most antagonistic starting points, which is the")
    say("  signature of regression to the mean in the measured data itself.")
    say("")

    say("=" * 78)
    say("  VERDICT")
    say("=" * 78)
    say("")
    diff = g_m.original_model_synergy.mean() - g_o.original_model_synergy.mean()
    pm = pool[pool.measured]
    po = pool[~pool.measured]
    sm = 100 * pm.pool_mean.mean() / pm.pool_max.mean()
    so = 100 * po.pool_mean.mean() / po.pool_max.mean()
    if diff < -5:
        say("  The 96 start %.1f Loewe units MORE antagonistic than the rest" % abs(diff))
        say("  (p=%.3g). Their larger measured gain is therefore partly a" % t.pvalue)
        say("  consequence of where they start, not of better selection.")
    elif diff > 5:
        say("  The 96 start %.1f units LESS antagonistic than the rest." % diff)
    else:
        say("  Baselines are comparable (%.1f units apart); the subset is not" % abs(diff))
        say("  selected on antagonism.")
    say("")
    say("  RTM share of gain: %.0f%% for the measured 96 vs %.0f%% elsewhere."
        % (sm, so))
    if sm > so + 3:
        say("  The two biases COMPOUND: the 96 are both easier cases and more")
        say("  regression-prone. The +50.99 headline needs both caveats.")
    elif sm < so - 3:
        say("  RTM explains LESS for the 96 than elsewhere, which strengthens")
        say("  rather than weakens the measured result.")
    else:
        say("  RTM contributes comparably in both strata, so the subset's")
        say("  larger gain is not additionally inflated by regression.")
    say("")

    with open(os.path.join(OUT, "e2_subset_bias.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("written -> %s/e2_subset_bias.txt" % OUT)


if __name__ == "__main__":
    main()
