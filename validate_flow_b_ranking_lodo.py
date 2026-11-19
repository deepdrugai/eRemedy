"""
validate_flow_b_ranking_lodo.py
===============================
Out-of-sample version of the Flow B ranking test.

validate_flow_b_ranking.py scored every candidate with the full teacher, which
had already seen those combinations, so its Spearman of 0.87 is an in-sample
number and cannot answer the circularity objection.

Here each candidate is scored by the leave-one-drug-out model that had THAT
candidate removed from training. Within a (cell line, anchor drug) context the
model is therefore choosing among drugs it has never seen combined with
anything -- which is exactly the situation Flow B is in when it proposes a
repurposing candidate from dataset_EVERYTHING.

Comparing the two gives the honest cost of extrapolation.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lodo-dir", default="clean_runs/lodo")
    p.add_argument("--insample", default="thesis/aggregate/flow_b_ranking_validation.csv")
    p.add_argument("--min-context", type=int, default=6)
    p.add_argument("--outdir", default="thesis/aggregate")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.lodo_dir, "shard_*", "predictions_*.csv")))
    if not files:
        raise SystemExit("No LODO prediction files under %s" % args.lodo_dir)
    print("loading %d LODO prediction files" % len(files))

    frames = []
    for f in files:
        drug = os.path.basename(f)[len("predictions_"):-len(".csv")]
        d = pd.read_csv(f)
        d["held_out"] = drug
        frames.append(d)
    lodo = pd.concat(frames, ignore_index=True)

    for c in ("cell_line", "drug_a", "drug_b", "held_out"):
        lodo[c] = lodo[c].astype(str).str.strip()

    # Collapse dose-level replicates to one row per (cell, pair).
    agg = lodo.groupby(["held_out", "cell_line", "drug_a", "drug_b"], as_index=False).agg(
        y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))

    # Orient each row so `cand` is the held-out drug and `anchor` is its partner.
    a_is_held = agg.drug_a == agg.held_out
    agg["cand"] = np.where(a_is_held, agg.drug_a, agg.drug_b)
    agg["anchor"] = np.where(a_is_held, agg.drug_b, agg.drug_a)
    agg = agg[agg.cand == agg.held_out]

    print("out-of-sample (cell, anchor, candidate) rows: %s" % format(len(agg), ","))

    rows = []
    for (cell, anchor), g in agg.groupby(["cell_line", "anchor"]):
        if len(g) < args.min_context:
            continue
        pred = g.y_pred.values
        obs = g.y_true.values
        names = g.cand.values
        if np.std(pred) == 0:
            continue
        pick = int(pred.argmax())
        best = int(obs.argmax())
        rows.append({
            "cell_line": cell,
            "anchor_drug": anchor,
            "n_candidates": len(g),
            "spearman": spearmanr(pred, obs).correlation,
            "pearson": np.corrcoef(pred, obs)[0, 1],
            "model_pick": names[pick],
            "true_best": names[best],
            "picked_observed": obs[pick],
            "best_observed": obs[best],
            "regret": obs[best] - obs[pick],
            "pick_in_top_quartile": bool(obs[pick] >= np.percentile(obs, 75)),
            "pick_is_best": names[pick] == names[best],
        })

    res = pd.DataFrame(rows).dropna(subset=["spearman"])
    out = os.path.join(args.outdir, "flow_b_ranking_validation_lodo.csv")
    res.to_csv(out, index=False)

    rng = np.random.default_rng(0)
    rr, rq = [], []
    for (cell, anchor), g in agg.groupby(["cell_line", "anchor"]):
        if len(g) < args.min_context:
            continue
        o = g.y_true.values
        j = rng.integers(len(o))
        rr.append(o.max() - o[j])
        rq.append(o[j] >= np.percentile(o, 75))

    print()
    print("=" * 88)
    print("  FLOW B RANKING -- OUT OF SAMPLE (each candidate scored by the model")
    print("  that had that candidate held out of training)")
    print("=" * 88)
    print("contexts evaluated            : %d" % len(res))
    print("mean candidates per context   : %.1f" % res.n_candidates.mean())
    print()
    print("Spearman(predicted, observed) : mean %+.3f | median %+.3f"
          % (res.spearman.mean(), res.spearman.median()))
    print("Pearson (predicted, observed) : mean %+.3f" % res.pearson.mean())
    print("contexts with rho > 0         : %.1f%%" % (100.0 * (res.spearman > 0).mean()))
    print("contexts with rho > 0.5       : %.1f%%" % (100.0 * (res.spearman > 0.5).mean()))
    print()
    print("top-1 is the true best        : %.1f%%  (chance %.1f%%)"
          % (100.0 * res.pick_is_best.mean(), 100.0 / res.n_candidates.mean()))
    print("top-1 in observed top quartile: %.1f%%  (chance 25%%)"
          % (100.0 * res.pick_in_top_quartile.mean()))
    print("mean regret (Loewe units)     : %.2f" % res.regret.mean())
    print("median regret                 : %.2f" % res.regret.median())
    print()
    print("random picker  regret         : %.2f" % np.mean(rr))
    print("random picker  top quartile   : %.1f%%" % (100.0 * np.mean(rq)))

    if os.path.exists(args.insample):
        ins = pd.read_csv(args.insample)
        print()
        print("-" * 88)
        print("  IN-SAMPLE vs OUT-OF-SAMPLE")
        print("-" * 88)
        print("  %-28s %14s %14s" % ("metric", "in-sample", "out-of-sample"))
        print("  " + "-" * 58)
        print("  %-28s %14.3f %14.3f" % ("mean Spearman",
                                         ins.spearman.mean(), res.spearman.mean()))
        print("  %-28s %13.1f%% %13.1f%%" % ("top-1 = true best",
                                             100 * ins.pick_is_best.mean(),
                                             100 * res.pick_is_best.mean()))
        print("  %-28s %13.1f%% %13.1f%%" % ("top-1 in top quartile",
                                             100 * ins.pick_in_top_quartile.mean(),
                                             100 * res.pick_in_top_quartile.mean()))
        print("  %-28s %14.2f %14.2f" % ("mean regret (Loewe)",
                                         ins.regret.mean(), res.regret.mean()))
    print("=" * 88)
    print("written -> %s" % out)


if __name__ == "__main__":
    main()
