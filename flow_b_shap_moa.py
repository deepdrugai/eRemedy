#!/usr/bin/env python3
"""
flow_b_shap_moa.py
==================
Flow B as designed: SHAP-driven, mechanism-matched drug replacement.

For a given cell line the pipeline is

  1. read the SHAP attribution table and take the fragments the teacher blames
     for lost synergy;
  2. for each such fragment, identify which of the two drugs actually carries
     it -- that drug is the replacement target, the other is retained;
  3. build a candidate pool of real drugs from dataset_EVERYTHING that share a
     mechanism of action with the target drug, so the substitution preserves
     therapeutic intent;
  4. score every candidate against the retained partner with the frozen teacher
     and report the change in predicted synergy.

This differs from an exhaustive sweep in three ways that matter. The pair and
the drug to be replaced are chosen by fragment-level attribution rather than by
raw antagonism, and mechanism of action defines the candidate pool rather than
annotating results after the fact.

Two controls are carried through because the earlier validation showed they are
needed. Substituting almost anything into a strongly antagonistic pair looks
like an improvement, so each candidate is reported as a z-score against the
distribution over its own pool, not only as a raw delta. And the teacher was
trained on 21 drugs, so every candidate is tagged with whether it falls inside
that vocabulary; external validation on NCI-ALMANAC put accuracy at PCC 0.47
inside it and 0.04 outside.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

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

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("flowB")

TEACHER_VOCAB = {
    "391210-10-9", "7-ethyl-10-hydroxycamptothecin", "915019-65-7", "bortezomib",
    "dasatinib", "dinaciclib", "mk-1775", "mk-2206", "mk-4541", "mrk003",
    "veliparib", "cyclophosphamide", "etoposide", "geldanamycin", "gemcitabine",
    "metformin", "methotrexate", "mitomycin c", "temozolomide", "vinblastine",
    "vinorelbine",
}


# --------------------------------------------------------------------------
# mechanism of action
# --------------------------------------------------------------------------
def load_moa_cache(path: str) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if not path or not os.path.exists(path):
        raise SystemExit("MoA cache not found: %s" % path)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("drug_name") or "").strip()
            if name:
                cache[name.lower()] = row
    n_moa = sum(1 for r in cache.values() if (r.get("mechanism_of_action") or "").strip())
    log.info("MoA cache: %d drugs (%d with mechanism text)", len(cache), n_moa)
    return cache


def target_set(row: dict) -> Set[str]:
    return {x.strip() for x in (row.get("target_chembl_ids") or "").split(";") if x.strip()}


def atc_l1_set(row: dict) -> Set[str]:
    return {c.strip()[0].upper()
            for c in (row.get("atc_classifications") or "").split(",") if c.strip()}


def moa_tokens(row: dict) -> Set[str]:
    txt = (row.get("mechanism_of_action") or "").lower().strip()
    return {txt} if txt else set()


def moa_match(target_row: dict, cand_row: dict) -> Tuple[bool, str]:
    """Return (is_match, confidence) for a candidate against the drug being replaced."""
    ta, tb = target_set(target_row), target_set(cand_row)
    if ta and tb and ta & tb:
        return True, "high"
    ma, mb = moa_tokens(target_row), moa_tokens(cand_row)
    if ma and mb and ma & mb:
        return True, "high"
    aa, ab = atc_l1_set(target_row), atc_l1_set(cand_row)
    if aa and ab and aa & ab:
        return True, "medium"
    return False, "none"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def pad_fragments(mats: Sequence[np.ndarray], device: str):
    n_max = max(m.shape[0] for m in mats)
    dim = mats[0].shape[1]
    out = np.zeros((len(mats), n_max, dim), dtype=np.float32)
    mask = np.zeros((len(mats), n_max), dtype=bool)
    for i, m in enumerate(mats):
        out[i, : m.shape[0]] = m
        mask[i, : m.shape[0]] = True
    return torch.from_numpy(out).to(device), torch.from_numpy(mask).to(device)


@torch.no_grad()
def score_pairs(model, cell_vec, partner_feat, cand_feats, device, bs=256) -> np.ndarray:
    """
    Score (candidate, partner) for every candidate, symmetrised over argument
    order because the model is symmetry-regularised but not exactly symmetric.
    """
    p_mol, p_frag = partner_feat
    out = np.zeros(len(cand_feats), dtype=np.float32)
    cell_t = torch.from_numpy(np.asarray(cell_vec, dtype=np.float32)).to(device)
    p_mol_t = torch.from_numpy(np.asarray(p_mol, dtype=np.float32)).to(device)

    for s in range(0, len(cand_feats), bs):
        chunk = cand_feats[s : s + bs]
        b = len(chunk)
        cells = cell_t.unsqueeze(0).expand(b, *cell_t.shape).contiguous()
        pm = p_mol_t.unsqueeze(0).expand(b, *p_mol_t.shape).contiguous()
        pf, pmask = pad_fragments([np.asarray(p_frag, dtype=np.float32)] * b, device)
        cm = torch.from_numpy(np.stack([c[0] for c in chunk]).astype(np.float32)).to(device)
        cf, cmask = pad_fragments([c[1] for c in chunk], device)

        p1 = model(cells, cm, pm, cf, cmask, pf, pmask).detach().cpu().numpy().ravel()
        p2 = model(cells, pm, cm, pf, pmask, cf, cmask).detach().cpu().numpy().ravel()
        out[s : s + b] = 0.5 * (p1 + p2)
    return out


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Flow B: SHAP-driven mechanism-matched replacement")
    p.add_argument("--shap-harmful", required=True,
                   help="top_200_harmful_fragments.csv for this cell line.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl")
    p.add_argument("--cell-line", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--moa-cache", default="drug_moa_cache_full.csv")
    p.add_argument("--molecule-embeddings", default=DEFAULT_MOLECULE_EMBEDDINGS)
    p.add_argument("--fragment-embeddings", default=DEFAULT_FRAGMENT_EMBEDDINGS)
    p.add_argument("--fragment-map", default=DEFAULT_FRAGMENT_MAP)
    p.add_argument("--drug-smiles-map", default=DEFAULT_DRUG_SMILES_MAP)
    p.add_argument("--top-fragments", type=int, default=50,
                   help="How many of the most harmful SHAP fragments to act on.")
    p.add_argument("--min-moa-confidence", choices=("high", "medium"), default="medium",
                   help="'high' requires a shared target or identical mechanism; "
                        "'medium' also allows a shared ATC level-1 class.")
    p.add_argument("--min-delta", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device
    cell = args.cell_line

    # ---- SHAP fragments ---------------------------------------------------
    shap = pd.read_csv(args.shap_harmful)
    need = {"drug_a", "drug_b", "drug_side", "fragment_smiles", "source_molecule_name",
            "other_molecule_name", "fragment_shap_value"}
    missing = need - set(shap.columns)
    if missing:
        raise SystemExit("SHAP table missing columns: %s" % sorted(missing))

    shap = shap.sort_values("fragment_shap_value").head(args.top_fragments)
    log.info("SHAP: acting on %d most harmful fragments (of %d)",
             len(shap), args.top_fragments)

    # ---- features ---------------------------------------------------------
    ck = load_checkpoint(Path(args.checkpoint), dev)
    model = build_model(ck, dev)
    model.eval()

    me = load_embedding_pickle(Path(args.molecule_embeddings))
    fe = load_embedding_pickle(Path(args.fragment_embeddings))
    fm = load_pickle(Path(args.fragment_map))
    dm = load_drug_smiles_map(Path(args.drug_smiles_map))
    smi_by_name = {str(k).strip().lower(): v for k, v in dm.items()}

    def feat(name: str):
        s = smi_by_name.get(str(name).strip().lower())
        if not s:
            return None
        mol = me.get(s)
        if mol is None:
            return None
        frag = _to_fragment_matrix(fm.get(s, []), fe)
        if frag is None:
            return None
        return (np.asarray(mol, dtype=np.float32), np.asarray(frag, dtype=np.float32))

    ds = load_pickle(Path(args.dataset))
    meta = ds["meta"]
    omics = ds["cell_omics"]
    sel = meta.cell_line_name.astype(str).str.strip().str.lower() == cell.strip().lower()
    if not sel.any():
        raise SystemExit("Cell line %s not in dataset" % cell)
    cell_vec = np.asarray(omics[int(np.flatnonzero(sel.values)[0])], dtype=np.float32)

    # observed synergy per measured pair, for later reference
    mm = meta[sel].assign(_a=meta[sel].drug_a.astype(str).str.strip(),
                          _b=meta[sel].drug_b.astype(str).str.strip())
    observed = {frozenset((a, b)): float(y) for a, b, y in
                zip(mm._a, mm._b, mm.y_loewe)}

    moa = load_moa_cache(args.moa_cache)

    # ---- candidate universe ----------------------------------------------
    universe = []
    for name in dm.keys():
        row = moa.get(str(name).strip().lower())
        if not row:
            continue
        f = feat(name)
        if f is None:
            continue
        universe.append((str(name).strip(), row, f))
    log.info("Candidate universe: %d encodable drugs with MoA annotation", len(universe))
    if not universe:
        raise SystemExit("No annotated candidates -- is the MoA cache built?")

    rank = {"high": 0, "medium": 1}
    allowed = {"high"} if args.min_moa_confidence == "high" else {"high", "medium"}

    rows: List[dict] = []
    skipped = {"no_moa_for_target": 0, "no_features": 0, "no_candidates": 0}
    seen_targets = set()

    for r in shap.itertuples(index=False):
        carrier = str(r.source_molecule_name).strip()
        partner = str(r.other_molecule_name).strip()
        key = (carrier, partner, str(r.fragment_smiles))
        if key in seen_targets:
            continue
        seen_targets.add(key)

        carrier_row = moa.get(carrier.lower())
        if not carrier_row:
            skipped["no_moa_for_target"] += 1
            continue
        cf, pf = feat(carrier), feat(partner)
        if cf is None or pf is None:
            skipped["no_features"] += 1
            continue

        # mechanism-matched candidates, excluding the two drugs already present
        cands = []
        for name, crow, cfeat in universe:
            if name.lower() in (carrier.lower(), partner.lower()):
                continue
            ok, conf = moa_match(carrier_row, crow)
            if ok and conf in allowed:
                cands.append((name, crow, cfeat, conf))
        if not cands:
            skipped["no_candidates"] += 1
            continue

        s_orig = float(score_pairs(model, cell_vec, pf, [cf], dev, args.batch_size)[0])
        preds = score_pairs(model, cell_vec, pf,
                            [c[2] for c in cands], dev, args.batch_size)
        deltas = preds - s_orig
        mu, sd = float(deltas.mean()), float(deltas.std()) or 1e-6
        order = deltas.argsort().argsort()

        frag_mol = Chem.MolFromSmiles(str(r.fragment_smiles))
        for i, (name, crow, _, conf) in enumerate(cands):
            csmi = smi_by_name.get(name.lower())
            still_has = None
            if frag_mol is not None and csmi:
                cm = Chem.MolFromSmiles(csmi)
                still_has = bool(cm.HasSubstructMatch(frag_mol)) if cm else None
            rows.append({
                "cell_line": cell,
                "drug_a": r.drug_a,
                "drug_b": r.drug_b,
                "harmful_fragment": r.fragment_smiles,
                "fragment_shap_value": float(r.fragment_shap_value),
                "fragment_side": r.drug_side,
                "replaced_drug": carrier,
                "retained_partner": partner,
                "candidate_drug": name,
                "candidate_smiles": csmi,
                "moa_confidence": conf,
                "replaced_moa": carrier_row.get("mechanism_of_action", ""),
                "candidate_moa": crow.get("mechanism_of_action", ""),
                "candidate_max_phase": crow.get("max_phase", ""),
                "original_model_synergy": s_orig,
                "repurposed_synergy": float(preds[i]),
                "delta_synergy": float(deltas[i]),
                "z_vs_moa_pool": float((deltas[i] - mu) / sd),
                "pct_rank_in_pool": float(order[i] / max(len(deltas) - 1, 1)),
                "pool_size": len(cands),
                "pool_mean_delta": mu,
                "candidate_retains_harmful_fragment": still_has,
                "candidate_in_teacher_vocab": name.lower() in TEACHER_VOCAB,
                "observed_loewe_original": observed.get(frozenset((carrier, partner)), np.nan),
                "observed_loewe_recommendation": observed.get(frozenset((name, partner)), np.nan),
            })

        log.info("  %s + %s | replace %s (%s) | %d MoA candidates | best delta %+.1f",
                 r.drug_a, r.drug_b, carrier, str(r.fragment_smiles)[:22],
                 len(cands), deltas.max())

    if not rows:
        raise SystemExit("No results. skipped=%s" % skipped)

    df = pd.DataFrame(rows)
    log.info("Scored %s mechanism-matched substitutions (skipped %s)",
             format(len(df), ","), skipped)

    hits = df[(df.delta_synergy >= args.min_delta)].copy()
    # drop candidates that reintroduce the very fragment SHAP flagged
    hits = hits[hits.candidate_retains_harmful_fragment != True]  # noqa: E712
    hits["_c"] = hits.moa_confidence.map(rank).fillna(9)
    hits = hits.sort_values(["_c", "z_vs_moa_pool"], ascending=[True, False]).drop(columns="_c")

    base = os.path.join(args.outdir, "flowB_shap_moa")
    df.to_csv("%s_all_%s.csv" % (base, cell), index=False)
    hits.to_csv("%s_hits_%s.csv" % (base, cell), index=False)
    write_summary("%s_summary_%s.txt" % (base, cell), df, hits, args, skipped)

    log.info("Hits: %d (MoA-high %d, in-vocab %d) -> %s",
             len(hits), int((hits.moa_confidence == "high").sum()),
             int(hits.candidate_in_teacher_vocab.sum()), args.outdir)


def write_summary(path, df, hits, args, skipped):
    L = []
    L.append("=" * 88)
    L.append("  FLOW B -- SHAP-DRIVEN, MECHANISM-MATCHED DRUG REPLACEMENT")
    L.append("  cell line: %s" % args.cell_line)
    L.append("=" * 88)
    L.append("")
    L.append("Pipeline: SHAP names the harmful fragment -> the drug carrying it is the")
    L.append("replacement target -> candidates are real drugs sharing its mechanism ->")
    L.append("the frozen teacher scores each against the retained partner.")
    L.append("")
    L.append("Fragment/target sites acted on   : %d" % df.groupby(
        ["replaced_drug", "retained_partner", "harmful_fragment"]).ngroups)
    L.append("Mechanism-matched substitutions  : %s" % format(len(df), ","))
    L.append("Mean candidates per site         : %.1f" % df.pool_size.mean())
    L.append("MoA stringency                   : %s" % args.min_moa_confidence)
    L.append("Sites skipped                    : %s" % skipped)
    L.append("")
    L.append("Hits (delta >= %.1f, fragment removed) : %d" % (args.min_delta, len(hits)))
    L.append("    mechanism match high         : %d" % int((hits.moa_confidence == "high").sum()))
    L.append("    mechanism match medium       : %d" % int((hits.moa_confidence == "medium").sum()))
    L.append("    inside teacher vocabulary    : %d" % int(hits.candidate_in_teacher_vocab.sum()))
    L.append("")

    L.append("-" * 88)
    L.append("CONTROLS")
    L.append("-" * 88)
    L.append("Regression to the mean: substituting anything into a strongly antagonistic")
    L.append("pair raises the predicted score, so the z-score against each site's own")
    L.append("candidate pool is the statistic to read, not the raw delta.")
    L.append("  mean delta over all MoA candidates : %+.2f" % df.delta_synergy.mean())
    L.append("  mean delta among hits              : %+.2f" % (hits.delta_synergy.mean()
                                                               if len(hits) else float("nan")))
    L.append("")
    L.append("Teacher reliability: trained on 21 drugs. External validation on NCI-ALMANAC")
    L.append("gives PCC 0.47 when both drugs are in that vocabulary and 0.04 when they are")
    L.append("not, so candidates outside it are screening hypotheses rather than estimates.")
    iv = hits[hits.candidate_in_teacher_vocab] if len(hits) else hits
    L.append("  hits inside vocabulary  : %d" % len(iv))
    L.append("  hits outside vocabulary : %d" % (len(hits) - len(iv)))
    L.append("")

    col = "observed_loewe_recommendation"
    if col in hits.columns and hits[col].notna().any():
        chk = hits.dropna(subset=[col, "observed_loewe_original"])
        if len(chk):
            gain = chk[col] - chk.observed_loewe_original
            L.append("-" * 88)
            L.append("GROUND TRUTH WHERE THE RECOMMENDED COMBINATION WAS MEASURED")
            L.append("-" * 88)
            L.append("  recommendations with a measured outcome : %d" % len(chk))
            L.append("  improvement realised experimentally     : %.1f%%" % (100.0 * (gain > 0).mean()))
            L.append("  mean realised gain                      : %+.2f" % gain.mean())
            L.append("  mean predicted gain                     : %+.2f" % chk.delta_synergy.mean())
            L.append("")

    L.append("-" * 88)
    L.append("TOP RECOMMENDATIONS")
    L.append("-" * 88)
    if hits.empty:
        L.append("  none passed the gates")
    else:
        for i, h in enumerate(hits.head(40).itertuples(index=False), 1):
            tag = "in-vocab" if h.candidate_in_teacher_vocab else "out-of-vocab"
            L.append("%3d. %s + %s" % (i, h.drug_a, h.drug_b))
            L.append("     harmful fragment : %s  (SHAP %.1f)"
                     % (str(h.harmful_fragment)[:44], h.fragment_shap_value))
            L.append("     replace %s -> %s  [%s match, %s]"
                     % (h.replaced_drug, h.candidate_drug, h.moa_confidence, tag))
            L.append("     keep %s | synergy %+.2f -> %+.2f (delta %+.2f, z=%.2f)"
                     % (h.retained_partner, h.original_model_synergy,
                        h.repurposed_synergy, h.delta_synergy, h.z_vs_moa_pool))
            if h.candidate_moa:
                L.append("     mechanism : %s" % str(h.candidate_moa)[:66])
            L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
