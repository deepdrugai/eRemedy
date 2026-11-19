"""
E7 -- Statistical rigour on the headline numbers
================================================
Four checks, all pure analysis on existing artefacts:

  A  Bootstrap CI on the 63% regression-to-the-mean share (1,361 sites).
  B  Fisher CI on r = 0.902 for the 96 measurement-backed recommendations,
     plus a bootstrap CI on the realised improvement rate.
  C  Benjamini-Hochberg FDR across the seven testable fragment tests, since
     "two of seven significantly reversed" is an uncorrected claim.
  D  Stability of recurrence counts. Recurrence across independently processed
     cell lines is the paper's main evidential device, so it matters whether
     counts swing when the panel is resampled. Bootstrapping over cell lines
     gives this without retraining anything.

Read-only. Writes only into experiments/results/.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

A = "thesis/aggregate/"
POOLS = "clean_runs/flow_b_shap_moa/*/flowB_shap_moa_all_*.csv"
OUT = "experiments/results"
SITE = ["cell_line", "harmful_fragment", "replaced_drug", "retained_partner"]
VAL = "rescored_delta_synergy_vs_model"
KEY = ["source_fragment_smiles", "candidate_fragment_smiles"]
RNG = np.random.default_rng(0)
NBOOT = 5000


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, float)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


def main():
    import glob
    os.makedirs(OUT, exist_ok=True)
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 78)
    say("  E7  STATISTICAL RIGOUR ON THE HEADLINE NUMBERS")
    say("=" * 78)
    say("")

    # ---- A: RTM share -----------------------------------------------------
    say("-" * 78)
    say("A  Regression-to-the-mean share, bootstrap CI")
    say("-" * 78)
    allsc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(POOLS))],
                      ignore_index=True)
    pool = (allsc.groupby(SITE).delta_synergy
            .agg(pool_mean="mean", pool_max="max").reset_index())
    arb, sel = pool.pool_mean.values, pool.pool_max.values
    point = 100.0 * arb.mean() / sel.mean()

    n = len(pool)
    boots = np.empty(NBOOT)
    for i in range(NBOOT):
        s = RNG.integers(0, n, n)
        boots[i] = 100.0 * arb[s].mean() / sel[s].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    say("  sites                          : %s" % format(n, ","))
    say("  arbitrary substitution         : %+.2f Loewe" % arb.mean())
    say("  model-selected candidate       : %+.2f Loewe" % sel.mean())
    say("  attributable to model          : %+.2f Loewe" % (sel.mean() - arb.mean()))
    say("  RTM share                      : %.1f%%  95%% CI [%.1f, %.1f]"
        % (point, lo, hi))
    say("")

    dboot = np.empty(NBOOT)
    for i in range(NBOOT):
        s = RNG.integers(0, n, n)
        dboot[i] = sel[s].mean() - arb[s].mean()
    say("  model advantage                : %+.2f  95%% CI [%+.2f, %+.2f]"
        % (sel.mean() - arb.mean(), *np.percentile(dboot, [2.5, 97.5])))
    say("")

    # ---- B: the 96 --------------------------------------------------------
    say("-" * 78)
    say("B  The 96 measurement-backed recommendations")
    say("-" * 78)
    mm = pd.read_csv(A + "flow_b_measured_recommendations.csv")
    r, p = stats.pearsonr(mm.delta_synergy, mm.realised_gain)
    k = len(mm)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(k - 3)
    ci = np.tanh([z - 1.96 * se, z + 1.96 * se])
    say("  n                              : %d" % k)
    say("  predicted vs realised          : r = %+.3f  95%% CI [%+.3f, %+.3f]"
        % (r, ci[0], ci[1]))
    say("  p                              : %.3g" % p)

    g = mm.realised_gain.values
    rate = 100.0 * (g > 0).mean()
    rb = np.array([100.0 * (g[RNG.integers(0, k, k)] > 0).mean()
                   for _ in range(NBOOT)])
    say("  realised improvement rate      : %.1f%%  95%% CI [%.1f, %.1f]"
        % (rate, *np.percentile(rb, [2.5, 97.5])))
    gb = np.array([g[RNG.integers(0, k, k)].mean() for _ in range(NBOOT)])
    say("  mean realised gain             : %+.2f  95%% CI [%+.2f, %+.2f]"
        % (g.mean(), *np.percentile(gb, [2.5, 97.5])))
    say("")
    say("  Note: a 100% rate with n=96 has an exact one-sided 95% lower bound")
    say("  of %.1f%% (rule of three), which is the honest floor to quote."
        % (100 * (1 - 3.0 / k)))
    say("")

    # ---- C: FDR on the fragment tests ------------------------------------
    say("-" * 78)
    say("C  Fragment-versus-measurement tests, FDR corrected")
    say("-" * 78)
    fv = pd.read_csv(A + "fragment_measurement_validation.csv")
    harm = fv[fv.set == "harmful"].copy()
    harm["p_adj"] = bh_fdr(harm.p.values)
    say("  testable harmful fragments     : %d" % len(harm))
    say("")
    say("  %-30s %9s %8s %10s %10s" % ("fragment", "effect", "d", "p", "p_adj"))
    say("  " + "-" * 72)
    for _, x in harm.sort_values("effect").iterrows():
        say("  %-30s %+9.2f %+8.3f %10.2e %10.2e"
            % (str(x.fragment)[:30], x.effect, x.cohens_d, x.p, x.p_adj))
    say("")
    sig = harm[harm.p_adj < 0.05]
    say("  significant after BH-FDR       : %d of %d" % (len(sig), len(harm)))
    say("     confirmed harmful (effect<0): %d" % int((sig.effect < 0).sum()))
    say("     reversed        (effect>0)  : %d" % int((sig.effect > 0).sum()))
    say("")

    # ---- D: recurrence stability -----------------------------------------
    say("-" * 78)
    say("D  Stability of recurrence counts under cell-line resampling")
    say("-" * 78)
    rp = pd.read_csv(A + "rescored_pooled.csv")
    cells = rp.cell_line_name.unique()
    nc = len(cells)
    swap_cells = (rp.groupby(KEY + ["cell_line_name"])[VAL].mean()
                    .reset_index())
    top = (swap_cells.groupby(KEY).cell_line_name.count()
           .sort_values(ascending=False).head(6))

    say("  resampling the %d cell lines with replacement, %d iterations"
        % (nc, 1000))
    say("")
    say("  %-42s %8s %16s" % ("swap", "observed", "95% CI"))
    say("  " + "-" * 70)
    rows = []
    for (src, cand), obs in top.items():
        sub = swap_cells[(swap_cells.source_fragment_smiles == src) &
                         (swap_cells.candidate_fragment_smiles == cand)]
        present = set(sub.cell_line_name)
        counts = np.array([sum(c in present for c in
                               cells[RNG.integers(0, nc, nc)])
                           for _ in range(1000)])
        lo_, hi_ = np.percentile(counts, [2.5, 97.5])
        lab = "%s -> %s" % (str(src)[:20], str(cand)[:16])
        say("  %-42s %8d %16s" % (lab, obs, "[%.0f, %.0f]" % (lo_, hi_)))
        rows.append(dict(source=src, candidate=cand, observed=obs,
                         ci_lo=lo_, ci_hi=hi_, sd=counts.std()))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "e7_recurrence_stability.csv"), index=False)
    say("")
    say("  mean bootstrap SD of recurrence count: %.2f cell lines" % R.sd.mean())
    say("")
    if R.sd.mean() < 2.0:
        say("  Recurrence counts are stable to about +/-%.0f cell lines, so the"
            % np.ceil(R.sd.mean() * 2))
        say("  paper can lean on them as its primary evidential device.")
    else:
        say("  Recurrence counts vary by more than +/-4 cell lines under")
        say("  resampling, which weakens their use as primary evidence.")
    say("")

    with open(os.path.join(OUT, "e7_statistics.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    harm.to_csv(os.path.join(OUT, "e7_fragment_fdr.csv"), index=False)
    print("written -> %s/e7_statistics.txt" % OUT)


if __name__ == "__main__":
    main()
