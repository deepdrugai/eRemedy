"""
E4 -- Is the cross-cell-line sign divergence real, or estimation noise?
=======================================================================
The paper reports that 34 of 42 fragment swaps recovered in >=15 cell lines
(81%) improve synergy in some lines and worsen it in others, and gives this
finding considerable prominence. Some per-cell-line means rest on a single
candidate, so a sign flip between two thin estimates may be noise rather than
biology.

Three checks:

  A  Replication filter -- recompute the divergence rate keeping only cell
     lines with at least k candidates supporting the estimate.
  B  Permutation test -- shuffle cell-line labels within each swap to build a
     null for how often sign divergence arises by chance.
  C  Variance decomposition -- how much of the spread in a swap's effect is
     between cell lines versus within a cell line.

Read-only. Writes only into experiments/results/.
"""

import os

import numpy as np
import pandas as pd

OUT = "experiments/results"
POOLED = "thesis/aggregate/rescored_pooled.csv"
VAL = "rescored_delta_synergy_vs_model"
KEY = ["source_fragment_smiles", "candidate_fragment_smiles"]
RNG = np.random.default_rng(0)


def divergence_rate(df, min_lines, min_n):
    """Fraction of swaps that are sign-divergent, under a replication filter."""
    cell = (df.groupby(KEY + ["cell_line_name"])[VAL]
              .agg(m="mean", n="count").reset_index())
    cell = cell[cell.n >= min_n]
    agg = (cell.groupby(KEY)
             .agg(lines=("m", "count"),
                  npos=("m", lambda s: int((s > 0).sum())),
                  nneg=("m", lambda s: int((s < 0).sum())),
                  mean=("m", "mean"), mn=("m", "min"), mx=("m", "max"))
             .reset_index())
    agg = agg[agg.lines >= min_lines]
    if agg.empty:
        return agg, np.nan
    div = ((agg.npos > 0) & (agg.nneg > 0)).mean()
    return agg, div


def main():
    os.makedirs(OUT, exist_ok=True)
    r = pd.read_csv(POOLED)
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 78)
    say("  E4  IS THE SIGN DIVERGENCE REAL?")
    say("=" * 78)
    say("")
    say("candidates pooled: %s" % format(len(r), ","))
    say("")

    # ---- how thin are the per-cell-line estimates? ------------------------
    cell = (r.groupby(KEY + ["cell_line_name"])[VAL]
              .agg(m="mean", n="count").reset_index())
    say("-" * 78)
    say("0  How thin are the per-cell-line estimates?")
    say("-" * 78)
    say("  (swap, cell line) estimates : %s" % format(len(cell), ","))
    say("  candidates behind each      : mean %.2f | median %.0f"
        % (cell.n.mean(), cell.n.median()))
    for k in (1, 2, 3, 5):
        say("     supported by exactly %d candidate%s : %5d (%.0f%%)"
            % (k, " " if k == 1 else "s", int((cell.n == k).sum()),
               100 * (cell.n == k).mean()))
    say("")

    # ---- A: replication filter -------------------------------------------
    say("-" * 78)
    say("A  Divergence rate under a replication filter")
    say("-" * 78)
    say("  %-10s %-10s %8s %10s" % ("min cands", "min lines", "swaps", "divergent"))
    say("  " + "-" * 42)
    rowsA = []
    for min_n in (1, 2, 3, 5):
        for min_lines in (10, 15):
            agg, div = divergence_rate(r, min_lines, min_n)
            say("  %-10d %-10d %8d %9s"
                % (min_n, min_lines, len(agg),
                   "n/a" if np.isnan(div) else "%.0f%%" % (100 * div)))
            rowsA.append(dict(min_candidates=min_n, min_lines=min_lines,
                              n_swaps=len(agg),
                              divergent_pct=np.nan if np.isnan(div) else 100 * div))
    pd.DataFrame(rowsA).to_csv(os.path.join(OUT, "e4_replication_filter.csv"),
                               index=False)
    say("")
    say("  The paper's headline uses min candidates = 1, min lines = 15.")
    say("")

    # ---- B: permutation test ---------------------------------------------
    say("-" * 78)
    say("B  Permutation test against a no-cell-line-effect null")
    say("-" * 78)
    say("  Shuffle cell-line labels within each swap, preserving the number of")
    say("  candidates per line, then recount sign divergence. If divergence is")
    say("  driven by within-swap noise rather than cell-line biology, the")
    say("  shuffled rate should match the observed rate.")
    say("")

    agg_obs, div_obs = divergence_rate(r, 15, 1)
    keep = set(map(tuple, agg_obs[KEY].values))
    sub = r[[tuple(x) in keep for x in r[KEY].values]].copy()

    n_perm = 200
    null = []
    for _ in range(n_perm):
        s = sub.copy()
        s["cell_line_name"] = (s.groupby(KEY)["cell_line_name"]
                                .transform(lambda x: RNG.permutation(x.values)))
        _, d = divergence_rate(s, 15, 1)
        null.append(d)
    null = np.array([x for x in null if not np.isnan(x)])
    say("  observed divergence rate : %.1f%%" % (100 * div_obs))
    say("  permuted null            : mean %.1f%% | 2.5%% %.1f%% | 97.5%% %.1f%%"
        % (100 * null.mean(), 100 * np.percentile(null, 2.5),
           100 * np.percentile(null, 97.5)))
    p_emp = float(np.mean(null >= div_obs))
    say("  P(null >= observed)      : %.3f  (%d permutations)" % (p_emp, len(null)))
    say("")
    if p_emp > 0.05:
        say("  The observed rate is INSIDE the null. Sign divergence at this")
        say("  level arises from within-swap variability, not from a cell-line")
        say("  effect. The 81%% figure cannot be attributed to biology.")
    else:
        say("  The observed rate EXCEEDS the null. Divergence is not explained")
        say("  by within-swap noise alone.")
    say("")

    # ---- C: variance decomposition ---------------------------------------
    say("-" * 78)
    say("C  Between-cell-line vs within-cell-line variance")
    say("-" * 78)
    rowsC = []
    for k, g in r.groupby(KEY):
        cl = g.groupby("cell_line_name")[VAL]
        if cl.ngroups < 10:
            continue
        means = cl.mean()
        within = g.groupby("cell_line_name")[VAL].var().mean()
        between = means.var()
        rowsC.append(dict(source=k[0], candidate=k[1], lines=cl.ngroups,
                          between=between, within=within,
                          icc=between / (between + within)
                          if np.isfinite(within) and (between + within) > 0 else np.nan))
    C = pd.DataFrame(rowsC).dropna(subset=["icc"])
    C.to_csv(os.path.join(OUT, "e4_variance_decomposition.csv"), index=False)
    say("  swaps with >=10 cell lines: %d" % len(C))
    say("  ICC (share of variance that is between-cell-line):")
    say("     mean %.3f | median %.3f | >0.5 in %d swaps (%.0f%%)"
        % (C.icc.mean(), C.icc.median(), int((C.icc > 0.5).sum()),
           100 * (C.icc > 0.5).mean()))
    say("")
    say("  ICC near 0 means the cell line explains little of the spread and")
    say("  most variation is between candidates within a line.")
    say("")

    say("=" * 78)
    say("  VERDICT")
    say("=" * 78)
    say("")
    agg3, div3 = divergence_rate(r, 15, 3)
    say("  headline (>=1 candidate/line) : %.0f%% of %d swaps divergent"
        % (100 * div_obs, len(agg_obs)))
    if not np.isnan(div3):
        say("  with >=3 candidates/line      : %.0f%% of %d swaps divergent"
            % (100 * div3, len(agg3)))
    say("  permutation p                 : %.3f" % p_emp)
    say("  median ICC                    : %.3f" % C.icc.median())
    say("")
    if p_emp > 0.05 or C.icc.median() < 0.3:
        say("  The divergence section needs rewriting. The phenomenon may be")
        say("  real, but the present evidence does not separate it from noise,")
        say("  and the clinical warning should be stated as a caution about")
        say("  estimate stability rather than as a demonstrated cell-line effect.")
    else:
        say("  The divergence result survives. Report it with the replication")
        say("  filter and the permutation p-value alongside the headline rate.")
    say("")

    with open(os.path.join(OUT, "e4_divergence_noise.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("written -> %s/e4_divergence_noise.txt" % OUT)


if __name__ == "__main__":
    main()
