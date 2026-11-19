"""
E1a -- Pool composition audit: retrieval or selection?
======================================================
The paper's headline is that 96 Flow 2 recommendations coincide with
combinations already measured in O'Neil, and that all 96 improved. Because
those combinations sit inside the teacher's training distribution, the
selection may be retrieval of a memorised label rather than a judgement.

For each of the 96 sites this script classifies every scored pool member Y by
whether the triple (Y, retained partner B, cell line C) exists as a labelled
O'Neil row, and asks:

  Q1  How many labelled alternatives did each pool actually contain?
  Q2  Was the winner always one of them?
  Q3  Where did the best UNLABELLED candidate rank?
  Q4  Within a pool, does the prediction track the training label?

No model inference. Read-only on all existing artefacts.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = "experiments/results"
POOLS = "clean_runs/flow_b_shap_moa/%s/flowB_shap_moa_all_%s.csv"
MEASURED = "thesis/aggregate/flow_b_measured_recommendations.csv"
SITE = ["cell_line", "harmful_fragment", "replaced_drug", "retained_partner"]


def norm(x):
    return str(x).strip().lower()


def oneil_labels(path, cache):
    """(cell, frozenset(drug_a, drug_b)) -> y_loewe, cached to a small CSV."""
    if os.path.exists(cache):
        m = pd.read_csv(cache)
    else:
        ds = pickle.load(open(path, "rb"))
        m = ds["meta"][["cell_line_name", "drug_a", "drug_b", "y_loewe"]].copy()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        m.to_csv(cache, index=False)
    lab = {}
    for c, a, b, y in zip(m.cell_line_name, m.drug_a, m.drug_b, m.y_loewe):
        lab[(norm(c), frozenset((norm(a), norm(b))))] = float(y)
    return lab, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl")
    ap.add_argument("--cache", default="experiments/results/_oneil_meta.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    lab, meta = oneil_labels(a.dataset, a.cache)
    mm = pd.read_csv(MEASURED)
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 78)
    say("  E1a  POOL COMPOSITION AUDIT -- retrieval or selection?")
    say("=" * 78)
    say("")
    say("O'Neil labelled triples : %s" % format(len(lab), ","))
    say("measured recommendations: %d" % len(mm))
    say("")

    rows = []
    pool_cache = {}
    for r in mm.itertuples(index=False):
        cl = r.cell_line
        if cl not in pool_cache:
            f = POOLS % (cl, cl)
            pool_cache[cl] = pd.read_csv(f) if os.path.exists(f) else None
        allsc = pool_cache[cl]
        if allsc is None:
            continue
        pool = allsc[(allsc.cell_line == cl) &
                     (allsc.harmful_fragment == r.harmful_fragment) &
                     (allsc.replaced_drug == r.replaced_drug) &
                     (allsc.retained_partner == r.retained_partner)]
        if pool.empty:
            continue
        pool = pool.sort_values("delta_synergy", ascending=False).reset_index(drop=True)
        B = norm(r.retained_partner)
        key = [(norm(cl), frozenset((norm(y), B))) for y in pool.candidate_drug]
        pool = pool.assign(labelled=[k in lab for k in key],
                           label=[lab.get(k, np.nan) for k in key])

        unl = pool[~pool.labelled]
        winner_lab = bool(pool.labelled.iloc[0])
        # within-pool agreement between prediction and training label
        sub = pool[pool.labelled]
        rho = (spearmanr(sub.repurposed_synergy, sub.label).correlation
               if len(sub) >= 4 else np.nan)

        rows.append(dict(
            cell_line=cl, replaced=r.replaced_drug, retained=r.retained_partner,
            winner=r.candidate_drug, pool_size=len(pool),
            n_labelled=int(pool.labelled.sum()),
            pct_labelled=100.0 * pool.labelled.mean(),
            winner_labelled=winner_lab,
            rank_best_unlabelled=(int(unl.index[0]) + 1) if len(unl) else np.nan,
            delta_winner=float(pool.delta_synergy.iloc[0]),
            delta_best_unlabelled=(float(unl.delta_synergy.iloc[0])
                                   if len(unl) else np.nan),
            rho_pred_vs_label=rho,
            realised_gain=float(r.realised_gain)))

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "e1a_pool_composition.csv"), index=False)

    say("-" * 78)
    say("Q1  How many labelled alternatives per pool?")
    say("-" * 78)
    say("  sites analysed              : %d of %d" % (len(R), len(mm)))
    say("  mean pool size              : %.1f candidates" % R.pool_size.mean())
    say("  mean labelled per pool      : %.1f  (median %.0f, range %d-%d)"
        % (R.n_labelled.mean(), R.n_labelled.median(),
           R.n_labelled.min(), R.n_labelled.max()))
    say("  labelled share of pool      : %.1f%%" % R.pct_labelled.mean())
    say("")
    say("  distribution of labelled-candidate counts:")
    vc = R.n_labelled.value_counts().sort_index()
    for k, v in vc.items():
        say("     %2d labelled : %3d sites  %s" % (k, v, "#" * min(v, 46)))
    say("")

    say("-" * 78)
    say("Q2  Was the winner always a labelled candidate?")
    say("-" * 78)
    say("  winner labelled : %d / %d  (%.1f%%)"
        % (R.winner_labelled.sum(), len(R), 100 * R.winner_labelled.mean()))
    say("")

    say("-" * 78)
    say("Q3  Where did the best UNLABELLED candidate rank?")
    say("-" * 78)
    v = R.rank_best_unlabelled.dropna()
    say("  best unlabelled rank: median %.0f | mean %.1f | rank 1 in %d sites"
        % (v.median(), v.mean(), int((v == 1).sum())))
    gap = (R.delta_winner - R.delta_best_unlabelled).dropna()
    say("  predicted-gain gap, winner minus best unlabelled:")
    say("     mean %+.2f | median %+.2f | negative in %d sites"
        % (gap.mean(), gap.median(), int((gap < 0).sum())))
    say("")

    say("-" * 78)
    say("Q4  Within a pool, does prediction track the training label?")
    say("-" * 78)
    rr = R.rho_pred_vs_label.dropna()
    if len(rr):
        say("  sites with >=4 labelled members: %d" % len(rr))
        say("  Spearman(prediction, O'Neil label): mean %+.3f | median %+.3f"
            % (rr.mean(), rr.median()))
    else:
        say("  too few labelled members per pool to compute (needs >=4).")
        say("  This is itself the answer to Q1.")
    say("")

    say("-" * 78)
    say("Q5  Enrichment: are labelled candidates over-selected?")
    say("-" * 78)
    base = R.pct_labelled.mean()
    obs = 100 * R.winner_labelled.mean()
    say("  labelled share of pool (chance rate)  : %.1f%%" % base)
    say("  labelled share of WINNERS (observed)  : %.1f%%" % obs)
    say("  enrichment                            : %.1fx" % (obs / base))
    say("")
    say("  If the model ranked on chemistry alone, the winner would be a")
    say("  labelled candidate about %.0f%% of the time. It is labelled %.0f%%" % (base, obs))
    say("  of the time. Membership of the training set, not merit, is the")
    say("  dominant predictor of winning.")
    say("")

    say("=" * 78)
    say("  VERDICT")
    say("=" * 78)
    say("")
    ml = R.n_labelled.mean()
    wl = 100 * R.winner_labelled.mean()
    enr = obs / base
    say("  Enrichment of labelled candidates among winners: %.1fx" % enr)
    say("  Within-pool Spearman(prediction, training label): %+.3f"
        % R.rho_pred_vs_label.dropna().mean())
    say("")
    if ml <= 6 and wl > 95:
        say("  RETRIEVAL-DOMINATED. Pools contain a mean of %.1f labelled" % ml)
        say("  candidates out of %.0f, and the winner is labelled in %.1f%% of" %
            (R.pool_size.mean(), wl))
        say("  sites. The model is choosing among a handful of combinations it")
        say("  was trained on, so the 96-case result demonstrates agreement with")
        say("  measurement inside the training distribution, not out-of-sample")
        say("  predictive skill. The headline must be reworded, and E1c would")
        say("  only confirm what this already shows.")
    elif ml > 15 and wl > 95:
        say("  NON-TRIVIAL RANKING UNDER RECALL. Pools contain a mean of %.1f" % ml)
        say("  labelled candidates, so picking the right one is a real ordering")
        say("  task even if every option was seen in training. Worth stating,")
        say("  and E1c is now worth running to make selection out-of-sample.")
    else:
        say("  INTERMEDIATE. Mean %.1f labelled per pool, winner labelled in" % ml)
        say("  %.1f%% of sites. Run E1c before settling the wording." % wl)
    say("")

    with open(os.path.join(OUT, "e1a_pool_composition.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("written -> %s/e1a_pool_composition.{txt,csv}" % OUT)


if __name__ == "__main__":
    main()
