"""
validate_flow_b_ranking.py
==========================
Flow B claims that, for a fixed partner drug and cell line, the teacher can
rank candidate replacement drugs by how synergistic they would be. This script
tests that claim on ground truth.

For every (partner drug, cell line) context we take the in-vocabulary drugs
whose combination with that partner was actually measured in O'Neil, ask the
teacher to score each one, and compare the predicted ranking with the observed
Loewe ranking.

  Spearman rho   agreement between predicted and observed ordering
  top-1 recovery does the model's first choice have an observed score in the
                 top quartile of that context
  regret         observed synergy of the true best partner minus that of the
                 model's pick, in Loewe units

This is the retrieval task Flow B performs, evaluated where the answer is
known. If ranking fails here it cannot be trusted on unseen drugs.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from synergy_prediction import (
    DEFAULT_DRUG_SMILES_MAP,
    DEFAULT_FRAGMENT_EMBEDDINGS,
    DEFAULT_FRAGMENT_MAP,
    DEFAULT_MOLECULE_EMBEDDINGS,
    _to_fragment_matrix,
    load_drug_smiles_map,
    load_embedding_pickle,
    load_pickle,
)
from synergy_shap_analysis import build_model, load_checkpoint
from flow_b_drug_repurposing import score_single


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl")
    p.add_argument("--molecule-embeddings", default=DEFAULT_MOLECULE_EMBEDDINGS)
    p.add_argument("--fragment-embeddings", default=DEFAULT_FRAGMENT_EMBEDDINGS)
    p.add_argument("--fragment-map", default=DEFAULT_FRAGMENT_MAP)
    p.add_argument("--drug-smiles-map", default=DEFAULT_DRUG_SMILES_MAP)
    p.add_argument("--min-context", type=int, default=6,
                   help="Minimum measured partners required to evaluate a context.")
    p.add_argument("--outdir", default="thesis/aggregate")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device

    ck = load_checkpoint(Path(args.checkpoint), dev)
    model = build_model(ck, dev)
    model.eval()

    ds = load_pickle(Path(args.dataset))
    meta = ds["meta"].copy()
    omics = np.asarray(ds["cell_omics"], dtype=np.float32)

    me = load_embedding_pickle(Path(args.molecule_embeddings))
    fe = load_embedding_pickle(Path(args.fragment_embeddings))
    fm = load_pickle(Path(args.fragment_map))
    dm = load_drug_smiles_map(Path(args.drug_smiles_map))
    low = {str(k).strip().lower(): v for k, v in dm.items()}

    feat = {}
    for d in set(meta.drug_a.astype(str).str.strip()) | set(meta.drug_b.astype(str).str.strip()):
        s = low.get(d.lower())
        if not s or me.get(s) is None:
            continue
        f = _to_fragment_matrix(fm.get(s, []), fe)
        if f is None:
            continue
        feat[d] = (np.asarray(me[s], dtype=np.float32), np.asarray(f, dtype=np.float32))
    print("encodable drugs: %d" % len(feat))

    meta["_row"] = np.arange(len(meta))
    meta["_a"] = meta.drug_a.astype(str).str.strip()
    meta["_b"] = meta.drug_b.astype(str).str.strip()
    meta["_c"] = meta.cell_line_name.astype(str).str.strip()
    m = meta[meta._a.isin(feat) & meta._b.isin(feat)]

    pairs = m.groupby(["_c", "_a", "_b"], as_index=False).agg(
        y=("y_loewe", "mean"), row=("_row", "first"))

    # symmetric long form: one row per (cell, anchor partner, candidate)
    long = pd.concat([
        pairs.rename(columns={"_a": "anchor", "_b": "cand"}),
        pairs.rename(columns={"_b": "anchor", "_a": "cand"}),
    ], ignore_index=True)

    results = []
    contexts = list(long.groupby(["_c", "anchor"]))
    print("evaluating %d (cell, anchor) contexts ..." % len(contexts))

    for i, ((cell, anchor), grp) in enumerate(contexts):
        if len(grp) < args.min_context:
            continue
        a_mol, a_frag = feat[anchor]
        cell_vec = omics[int(grp.row.iloc[0])]

        preds, obs, names = [], [], []
        for r in grp.itertuples(index=False):
            c_mol, c_frag = feat[r.cand]
            # average both argument orders; the model is symmetry-regularised
            # but not exactly symmetric
            p1 = score_single(model, cell_vec, a_mol, c_mol, a_frag, c_frag, dev)
            p2 = score_single(model, cell_vec, c_mol, a_mol, c_frag, a_frag, dev)
            preds.append(0.5 * (p1 + p2))
            obs.append(r.y)
            names.append(r.cand)

        preds = np.asarray(preds)
        obs = np.asarray(obs)
        rho = spearmanr(preds, obs).correlation
        pcc = np.corrcoef(preds, obs)[0, 1] if len(preds) > 2 else np.nan

        pick = int(preds.argmax())
        best = int(obs.argmax())
        q75 = np.percentile(obs, 75)

        results.append({
            "cell_line": cell,
            "anchor_drug": anchor,
            "n_candidates": len(grp),
            "spearman": rho,
            "pearson": pcc,
            "model_pick": names[pick],
            "true_best": names[best],
            "picked_observed": obs[pick],
            "best_observed": obs[best],
            "regret": obs[best] - obs[pick],
            "pick_in_top_quartile": bool(obs[pick] >= q75),
            "pick_is_best": names[pick] == names[best],
        })

        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, len(contexts)))

    res = pd.DataFrame(results)
    out = os.path.join(args.outdir, "flow_b_ranking_validation.csv")
    res.to_csv(out, index=False)

    print()
    print("=" * 88)
    print("  FLOW B RANKING VALIDATION  (in-vocabulary partners, ground truth known)")
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

    # A random-picker baseline over the same contexts.
    rng = np.random.default_rng(0)
    rand_regret, rand_topq = [], []
    for (cell, anchor), grp in long.groupby(["_c", "anchor"]):
        if len(grp) < args.min_context:
            continue
        o = grp.y.values
        j = rng.integers(len(o))
        rand_regret.append(o.max() - o[j])
        rand_topq.append(o[j] >= np.percentile(o, 75))
    print("random picker  regret         : %.2f" % np.mean(rand_regret))
    print("random picker  top quartile   : %.1f%%" % (100.0 * np.mean(rand_topq)))
    print("=" * 88)
    print("written -> %s" % out)


if __name__ == "__main__":
    main()
