#!/usr/bin/env python3
"""
validate_external_almanac.py
============================
External validation of the frozen O'Neil-trained teacher on NCI-ALMANAC.

The teacher was trained only on O'Neil. ALMANAC is an independent study that
shares the omics feature space but contributes 53 drugs the teacher has never
seen, which makes it a direct test of the generalisation that drug repurposing
depends on. Nothing is retrained; the checkpoint is used exactly as shipped.

Combinations are split into three strata so that study transfer and drug
generalisation can be told apart:

  both_known    both drugs are in the teacher's 21-drug training vocabulary.
                Performance here measures O'Neil -> ALMANAC transfer, i.e.
                assay and protocol differences, with no novel chemistry.
  one_unseen    exactly one partner is new.
  both_unseen   neither partner was seen in training. This is the repurposing
                regime.

A drop from both_known to both_unseen is drug-generalisation failure. Weak
performance in every stratum instead points at study batch effects and says
nothing about repurposing.

Two families of metric are reported. Pooled regression metrics (PCC, Spearman,
RMSE) describe absolute accuracy. Within-context ranking metrics describe the
task Flow B actually performs: holding a cell line and one partner fixed, can
the model order the candidate partners? Those are reported as PCC, Spearman,
top-1 accuracy, precision@3 and regret, each against a random-picker baseline.
"""

import argparse
import json
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

# The 21 O'Neil drugs the teacher can encode; everything else is unseen.
TRAIN_VOCAB = {
    "391210-10-9", "7-Ethyl-10-hydroxycamptothecin", "915019-65-7", "Bortezomib",
    "Dasatinib", "Dinaciclib", "MK-1775", "MK-2206", "MK-4541", "MRK003",
    "Veliparib", "cyclophosphamide", "etoposide", "geldanamycin", "gemcitabine",
    "metformin", "methotrexate", "mitomycin C", "temozolomide", "vinblastine",
    "vinorelbine",
}
TRAIN_LOWER = {d.lower() for d in TRAIN_VOCAB}

ONEIL_CELLS = {
    "A2058", "A2780", "A375", "A427", "CAOV3", "DLD1", "ES2", "HCT116", "HT144",
    "HT29", "KPL1", "LOVO", "MDAMB436", "MSTO", "NCI-H460", "NCIH1650",
    "NCIH2122", "NCIH23", "NCIH520", "OV90", "OVCAR3", "PA1", "RKO", "RPMI7951",
    "SK-OV-3", "SKMEL30", "SKMES1", "SW-620", "SW837", "T-47D", "UACC62",
    "UWB1289", "VCAP", "ZR751",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="clean_runs/2026-04-29_full_rerun/synergy/oneil_synergy_runs/model.pt")
    p.add_argument("--dataset", default="/media/exxact1/magesh/ecomb/dataset_ALMANAC.pkl")
    p.add_argument("--molecule-embeddings", default=DEFAULT_MOLECULE_EMBEDDINGS)
    p.add_argument("--fragment-embeddings", default=DEFAULT_FRAGMENT_EMBEDDINGS)
    p.add_argument("--fragment-map", default=DEFAULT_FRAGMENT_MAP)
    p.add_argument("--drug-smiles-map", default=DEFAULT_DRUG_SMILES_MAP)
    p.add_argument("--outdir", default="clean_runs/external_almanac")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-context", type=int, default=6,
                   help="Minimum candidates before a (cell, anchor) context is ranked.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pad_fragments(mats, device):
    n_max = max(m.shape[0] for m in mats)
    dim = mats[0].shape[1]
    out = np.zeros((len(mats), n_max, dim), dtype=np.float32)
    mask = np.zeros((len(mats), n_max), dtype=bool)
    for i, m in enumerate(mats):
        out[i, : m.shape[0]] = m
        mask[i, : m.shape[0]] = True
    return torch.from_numpy(out).to(device), torch.from_numpy(mask).to(device)


@torch.no_grad()
def score_all(model, omics, rows, feats, pair_a, pair_b, device, bs):
    """Symmetrised prediction for each (cell row, drug_a, drug_b) triple."""
    n = len(rows)
    preds = np.zeros(n, dtype=np.float32)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        cells = torch.from_numpy(
            np.asarray(omics[rows[s:e]], dtype=np.float32)).to(device)
        ma = torch.from_numpy(
            np.stack([feats[d][0] for d in pair_a[s:e]]).astype(np.float32)).to(device)
        mb = torch.from_numpy(
            np.stack([feats[d][0] for d in pair_b[s:e]]).astype(np.float32)).to(device)
        fa, mka = pad_fragments([feats[d][1] for d in pair_a[s:e]], device)
        fb, mkb = pad_fragments([feats[d][1] for d in pair_b[s:e]], device)

        p1 = model(cells, ma, mb, fa, mka, fb, mkb).detach().cpu().numpy().ravel()
        p2 = model(cells, mb, ma, fb, mkb, fa, mka).detach().cpu().numpy().ravel()
        preds[s:e] = 0.5 * (p1 + p2)

        if (s // bs) % 100 == 0:
            print("    scored %s / %s" % (format(e, ","), format(n, ",")), flush=True)
    return preds


def pooled_metrics(y_true, y_pred):
    if len(y_true) < 3:
        return {}
    return {
        "n": int(len(y_true)),
        "pearson": float(np.corrcoef(y_pred, y_true)[0, 1]),
        "spearman": float(spearmanr(y_pred, y_true).correlation),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_true_std": float(np.std(y_true)),
        "y_pred_std": float(np.std(y_pred)),
    }


def ranking_metrics(df, min_context, seed=0):
    """
    Within-context ranking. Each row of `df` is one (cell line, anchor, candidate)
    triple; contexts are (cell line, anchor). Returns per-context rows.
    """
    rng = np.random.default_rng(seed)
    out = []
    for (cell, anchor), g in df.groupby(["cell_line", "anchor"]):
        if len(g) < min_context:
            continue
        pred = g.y_pred.values
        obs = g.y_true.values
        if np.std(pred) == 0 or np.std(obs) == 0:
            continue
        names = g.cand.values
        k = min(3, len(g))
        top3_pred = set(names[np.argsort(-pred)[:k]])
        top3_true = set(names[np.argsort(-obs)[:k]])
        pick = int(pred.argmax())
        best = int(obs.argmax())
        j = int(rng.integers(len(obs)))
        out.append({
            "cell_line": cell,
            "anchor": anchor,
            "n_candidates": len(g),
            "pearson": float(np.corrcoef(pred, obs)[0, 1]),
            "spearman": float(spearmanr(pred, obs).correlation),
            "top1_is_best": bool(names[pick] == names[best]),
            "top1_in_top_quartile": bool(obs[pick] >= np.percentile(obs, 75)),
            "precision_at_3": len(top3_pred & top3_true) / float(k),
            "regret": float(obs[best] - obs[pick]),
            "random_regret": float(obs[best] - obs[j]),
            "random_in_top_quartile": bool(obs[j] >= np.percentile(obs, 75)),
            "chance_top1": 1.0 / len(g),
        })
    return pd.DataFrame(out)


def summarise_ranking(r):
    if r.empty:
        return {}
    return {
        "contexts": int(len(r)),
        "mean_candidates": float(r.n_candidates.mean()),
        "pearson_mean": float(r.pearson.mean()),
        "spearman_mean": float(r.spearman.mean()),
        "spearman_median": float(r.spearman.median()),
        "pct_contexts_rho_gt_0": float(100.0 * (r.spearman > 0).mean()),
        "top1_accuracy": float(100.0 * r.top1_is_best.mean()),
        "top1_chance": float(100.0 * r.chance_top1.mean()),
        "top1_in_top_quartile": float(100.0 * r.top1_in_top_quartile.mean()),
        "random_in_top_quartile": float(100.0 * r.random_in_top_quartile.mean()),
        "precision_at_3": float(100.0 * r.precision_at_3.mean()),
        "mean_regret": float(r.regret.mean()),
        "random_regret": float(r.random_regret.mean()),
    }


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device

    print("loading teacher -> %s" % args.checkpoint, flush=True)
    ck = load_checkpoint(Path(args.checkpoint), dev)
    model = build_model(ck, dev)
    model.eval()

    print("loading embeddings", flush=True)
    me = load_embedding_pickle(Path(args.molecule_embeddings))
    fe = load_embedding_pickle(Path(args.fragment_embeddings))
    fm = load_pickle(Path(args.fragment_map))
    dm = load_drug_smiles_map(Path(args.drug_smiles_map))
    low = {str(k).strip().lower(): v for k, v in dm.items()}

    print("loading ALMANAC -> %s" % args.dataset, flush=True)
    ds = load_pickle(Path(args.dataset))
    meta = ds["meta"]
    omics = ds["cell_omics"]

    m = meta.assign(
        _row=np.arange(len(meta)),
        _a=meta.drug_a.astype(str).str.strip(),
        _b=meta.drug_b.astype(str).str.strip(),
        _c=meta.cell_line_name.astype(str).str.strip(),
    )

    feats = {}
    for d in sorted(set(m._a) | set(m._b)):
        s = low.get(d.lower())
        if not s or me.get(s) is None:
            continue
        f = _to_fragment_matrix(fm.get(s, []), fe)
        if f is None:
            continue
        feats[d] = (np.asarray(me[s], dtype=np.float32), np.asarray(f, dtype=np.float32))

    unseen = {d for d in feats if d.lower() not in TRAIN_LOWER}
    print("encodable ALMANAC drugs: %d (%d unseen in training)"
          % (len(feats), len(unseen)), flush=True)

    m = m[m._a.isin(feats) & m._b.isin(feats)]
    combos = m.groupby(["_c", "_a", "_b"], as_index=False).agg(
        y_true=("y_loewe", "mean"), row=("_row", "first"))
    combos = combos.rename(columns={"_c": "cell_line", "_a": "drug_a", "_b": "drug_b"})
    print("unique (cell, pair) combinations: %s" % format(len(combos), ","), flush=True)

    print("scoring with frozen teacher ...", flush=True)
    combos["y_pred"] = score_all(
        model, omics, combos.row.values, feats,
        combos.drug_a.values, combos.drug_b.values, dev, args.batch_size)

    a_new = ~combos.drug_a.str.lower().isin(TRAIN_LOWER)
    b_new = ~combos.drug_b.str.lower().isin(TRAIN_LOWER)
    n_new = a_new.astype(int) + b_new.astype(int)
    combos["stratum"] = np.select(
        [n_new == 0, n_new == 1, n_new == 2],
        ["both_known", "one_unseen", "both_unseen"])
    combos["cell_seen_in_oneil"] = combos.cell_line.isin(ONEIL_CELLS)

    combos.to_csv(os.path.join(args.outdir, "almanac_predictions.csv"), index=False)

    # ---- long form for ranking: each pair contributes both orientations ----
    long = pd.concat([
        combos.rename(columns={"drug_a": "anchor", "drug_b": "cand"}),
        combos.rename(columns={"drug_b": "anchor", "drug_a": "cand"}),
    ], ignore_index=True)

    report = {"pooled": {}, "ranking": {}}

    print()
    print("=" * 100)
    print("  EXTERNAL VALIDATION -- FROZEN O'NEIL TEACHER ON NCI-ALMANAC")
    print("=" * 100)
    print("  encodable drugs %d (%d unseen) | combinations %s | cell lines %d"
          % (len(feats), len(unseen), format(len(combos), ","), combos.cell_line.nunique()))
    print()
    print("-" * 100)
    print("  POOLED REGRESSION ACCURACY BY STRATUM")
    print("-" * 100)
    print("  %-14s %8s %9s %9s %8s %8s %10s %10s"
          % ("stratum", "n", "PCC", "Spearman", "RMSE", "MAE", "true sd", "pred sd"))
    print("  " + "-" * 92)
    for st in ("both_known", "one_unseen", "both_unseen"):
        s = combos[combos.stratum == st]
        mt = pooled_metrics(s.y_true.values, s.y_pred.values)
        if not mt:
            continue
        report["pooled"][st] = mt
        print("  %-14s %8s %+9.3f %+9.3f %8.2f %8.2f %10.2f %10.2f"
              % (st, format(mt["n"], ","), mt["pearson"], mt["spearman"],
                 mt["rmse"], mt["mae"], mt["y_true_std"], mt["y_pred_std"]))
    mt = pooled_metrics(combos.y_true.values, combos.y_pred.values)
    report["pooled"]["all"] = mt
    print("  " + "-" * 92)
    print("  %-14s %8s %+9.3f %+9.3f %8.2f %8.2f %10.2f %10.2f"
          % ("ALL", format(mt["n"], ","), mt["pearson"], mt["spearman"],
             mt["rmse"], mt["mae"], mt["y_true_std"], mt["y_pred_std"]))

    # restricted to cell lines the teacher already knows
    print()
    print("-" * 100)
    print("  POOLED ACCURACY ON THE 9 CELL LINES SHARED WITH O'NEIL")
    print("  (cell context already learned, so only the chemistry is new)")
    print("-" * 100)
    print("  %-14s %8s %9s %9s %8s" % ("stratum", "n", "PCC", "Spearman", "RMSE"))
    print("  " + "-" * 52)
    for st in ("both_known", "one_unseen", "both_unseen"):
        s = combos[(combos.stratum == st) & combos.cell_seen_in_oneil]
        mt = pooled_metrics(s.y_true.values, s.y_pred.values)
        if not mt:
            continue
        report["pooled"]["shared_cells_" + st] = mt
        print("  %-14s %8s %+9.3f %+9.3f %8.2f"
              % (st, format(mt["n"], ","), mt["pearson"], mt["spearman"], mt["rmse"]))

    # ---- ranking ----------------------------------------------------------
    print()
    print("-" * 100)
    print("  WITHIN-CONTEXT RANKING  (fix cell line and one partner, order the candidates)")
    print("  This is the task Flow B performs.")
    print("-" * 100)
    hdr = ("  %-14s %8s %8s %9s %8s %8s %9s %8s %8s"
           % ("stratum", "contexts", "PCC", "Spearman", "top-1", "chance", "prec@3", "regret", "random"))
    print(hdr)
    print("  " + "-" * 92)

    rank_frames = {}
    for st in ("both_known", "one_unseen", "both_unseen", "all"):
        sub = long if st == "all" else long[long.stratum == st]
        r = ranking_metrics(sub, args.min_context)
        if r.empty:
            continue
        rank_frames[st] = r
        s = summarise_ranking(r)
        report["ranking"][st] = s
        print("  %-14s %8d %+8.3f %+9.3f %7.1f%% %7.1f%% %8.1f%% %8.2f %8.2f"
              % (st, s["contexts"], s["pearson_mean"], s["spearman_mean"],
                 s["top1_accuracy"], s["top1_chance"], s["precision_at_3"],
                 s["mean_regret"], s["random_regret"]))

    if rank_frames:
        pd.concat([v.assign(stratum=k) for k, v in rank_frames.items()],
                  ignore_index=True).to_csv(
            os.path.join(args.outdir, "almanac_ranking_contexts.csv"), index=False)

    with open(os.path.join(args.outdir, "almanac_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 100)
    print("  INTERPRETATION")
    print("=" * 100)
    pk = report["pooled"].get("both_known", {})
    pu = report["pooled"].get("both_unseen", {})
    if pk and pu:
        print("  transfer control (both drugs known) PCC : %+.3f" % pk["pearson"])
        print("  repurposing regime (both unseen)    PCC : %+.3f" % pu["pearson"])
        if pk["pearson"] < 0.2:
            print("  -> the teacher does not transfer to ALMANAC even on familiar drugs,")
            print("     so this stratum cannot separate batch effects from drug")
            print("     generalisation. Treat the unseen-drug numbers as uninformative.")
        elif pu["pearson"] < 0.5 * pk["pearson"]:
            print("  -> transfer works on familiar drugs but degrades sharply on new")
            print("     chemistry: genuine drug-generalisation failure.")
        else:
            print("  -> accuracy is retained on unseen chemistry; the teacher generalises")
            print("     to new drugs and open-pool repurposing is supportable.")
    rk = report["ranking"].get("both_unseen", {})
    if rk:
        print()
        print("  ranking unseen drugs: top-1 %.1f%% vs chance %.1f%%, regret %.2f vs random %.2f"
              % (rk["top1_accuracy"], rk["top1_chance"],
                 rk["mean_regret"], rk["random_regret"]))
    print("=" * 100)
    print("written -> %s" % args.outdir)


if __name__ == "__main__":
    main()
