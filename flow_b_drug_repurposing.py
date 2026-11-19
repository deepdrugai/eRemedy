#!/usr/bin/env python3
"""
flow_b_drug_repurposing.py
==========================
Flow B -- drug-level repurposing scored directly by the teacher model.

Flow A (the bioisostere pipeline) edits a single fragment of one partner and
scores the eSynth-reconstructed hypothetical molecule. Flow B instead takes an
antagonistic pair (D_a, D_b) that was measured in a given cell line and asks a
different question: if we swap out one partner for a REAL drug from
dataset_EVERYTHING, what does the teacher predict?

    S_orig  = f(D_a, D_b, cell)
    S_new   = f(D_alt, D_b, cell)          # D_a replaced
    dS      = S_new - S_orig

Every candidate in the pool is scored for every pair, which is what makes the
result interpretable. LODO diagnostics showed the teacher regresses unseen
drugs toward the training mean, so a large raw dS is expected for ANY
substitution into a strongly antagonistic pair. Scoring the full pool gives the
per-pair null distribution of dS for free, and each candidate is reported as a
z-score and percentile against that null. Those statistics are invariant to the
pair-level mean shift, so they measure candidate-specific signal rather than
regression to the mean.

Outputs
-------
  flowB_all_scores_<cell>.csv     every (pair, side, candidate) triple
  flowB_hits_<cell>.csv           candidates passing the z / delta / MoA gates
  flowB_null_stats_<cell>.csv     per-pair null distribution parameters
  flowB_summary_<cell>.txt        human-readable report
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("flowB")


# --------------------------------------------------------------------------
# candidate pool
# --------------------------------------------------------------------------
class Candidate:
    __slots__ = ("name", "smiles", "mol", "frag")

    def __init__(self, name, smiles, mol, frag):
        self.name = name
        self.smiles = smiles
        self.mol = mol
        self.frag = frag


def build_candidate_pool(
    drug_smiles_map: Dict[str, str],
    molecule_embeddings: Dict[str, np.ndarray],
    fragment_map: Dict[str, List[str]],
    fragment_embeddings: Dict[str, np.ndarray],
    max_fragments: Optional[int] = None,
) -> List[Candidate]:
    pool: List[Candidate] = []
    seen_smiles = set()
    skipped = {"no_embedding": 0, "no_fragments": 0, "duplicate": 0}

    for name, smiles in drug_smiles_map.items():
        if smiles in seen_smiles:
            skipped["duplicate"] += 1
            continue
        mol = molecule_embeddings.get(smiles)
        if mol is None:
            skipped["no_embedding"] += 1
            continue
        frag = _to_fragment_matrix(fragment_map.get(smiles, []), fragment_embeddings)
        if frag is None:
            skipped["no_fragments"] += 1
            continue
        if max_fragments is not None and frag.shape[0] > max_fragments:
            frag = frag[:max_fragments]
        seen_smiles.add(smiles)
        pool.append(Candidate(name,
                              smiles,
                              np.asarray(mol, dtype=np.float32),
                              np.asarray(frag, dtype=np.float32)))

    log.info("Candidate pool: %d drugs (skipped %s)", len(pool), skipped)
    return pool


# --------------------------------------------------------------------------
# batched scoring
# --------------------------------------------------------------------------
def _pad_fragments(mats: Sequence[np.ndarray], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (n_i, d) fragment matrices into (B, n_max, d) plus a mask."""
    n_max = max(m.shape[0] for m in mats)
    dim = mats[0].shape[1]
    out = np.zeros((len(mats), n_max, dim), dtype=np.float32)
    mask = np.zeros((len(mats), n_max), dtype=bool)
    for i, m in enumerate(mats):
        out[i, : m.shape[0]] = m
        mask[i, : m.shape[0]] = True
    return (torch.from_numpy(out).to(device),
            torch.from_numpy(mask).to(device))


@torch.no_grad()
def score_substitutions(
    model,
    cell: np.ndarray,
    partner_mol: np.ndarray,
    partner_frag: np.ndarray,
    candidates: Sequence[Candidate],
    replace_side: str,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Score every candidate substituted into `replace_side` of the pair while the
    partner occupies the other side. Returns an array of predictions aligned
    with `candidates`.
    """
    preds = np.zeros(len(candidates), dtype=np.float32)
    cell_t = torch.from_numpy(np.asarray(cell, dtype=np.float32)).to(device)
    partner_mol_t = torch.from_numpy(np.asarray(partner_mol, dtype=np.float32)).to(device)
    partner_frag_np = np.asarray(partner_frag, dtype=np.float32)

    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        b = len(chunk)

        cand_mol = torch.from_numpy(
            np.stack([c.mol for c in chunk]).astype(np.float32)).to(device)
        cand_frag, cand_mask = _pad_fragments([c.frag for c in chunk], device)

        p_frag, p_mask = _pad_fragments([partner_frag_np] * b, device)
        # cell omics is a (n_genes, n_modalities) matrix, not a flat vector, so
        # broadcast over whatever trailing shape each tensor happens to have.
        p_mol = partner_mol_t.unsqueeze(0).expand(b, *partner_mol_t.shape).contiguous()
        cells = cell_t.unsqueeze(0).expand(b, *cell_t.shape).contiguous()

        if replace_side == "a":
            out = model(cells, cand_mol, p_mol, cand_frag, cand_mask, p_frag, p_mask)
        else:
            out = model(cells, p_mol, cand_mol, p_frag, p_mask, cand_frag, cand_mask)

        preds[start : start + b] = out.detach().cpu().numpy().ravel()

    return preds


@torch.no_grad()
def score_single(model, cell, mol_a, mol_b, frag_a, frag_b, device) -> float:
    fa, ma = _pad_fragments([np.asarray(frag_a, dtype=np.float32)], device)
    fb, mb = _pad_fragments([np.asarray(frag_b, dtype=np.float32)], device)
    out = model(
        torch.from_numpy(np.asarray(cell, dtype=np.float32)[None, ...]).to(device),
        torch.from_numpy(np.asarray(mol_a, dtype=np.float32)[None, ...]).to(device),
        torch.from_numpy(np.asarray(mol_b, dtype=np.float32)[None, ...]).to(device),
        fa, ma, fb, mb,
    )
    return float(out.detach().cpu().numpy().ravel()[0])


# --------------------------------------------------------------------------
# MoA
# --------------------------------------------------------------------------
def load_moa_cache(csv_path: str) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if not csv_path or not os.path.exists(csv_path):
        log.warning("MoA cache not found (%s) -- proceeding unannotated", csv_path)
        return cache
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cache[row["drug_name"].strip().lower()] = row
    log.info("MoA cache: %d entries", len(cache))
    return cache


def _atc_l1(row: dict):
    codes = (row or {}).get("atc_classifications", "") or ""
    return {c.strip()[0].upper() for c in codes.split(",") if c.strip()}


def _targets(row: dict):
    t = (row or {}).get("target_chembl_ids", "") or ""
    return {x.strip() for x in t.split(";") if x.strip()}


def moa_relation(replaced: str, candidate: str, cache: Dict[str, dict]) -> dict:
    a = cache.get(str(replaced).strip().lower(), {})
    b = cache.get(str(candidate).strip().lower(), {})
    ta, tb = _targets(a), _targets(b)
    ca, cb = _atc_l1(a), _atc_l1(b)

    target_match = bool(ta and tb and ta & tb)
    atc_match = bool(ca and cb and ca & cb)

    if not a or not b:
        conf = "unknown"
    elif target_match:
        conf = "high"
    elif atc_match:
        conf = "medium"
    else:
        conf = "low"

    return {
        "replaced_moa": a.get("mechanism_of_action", ""),
        "candidate_moa": b.get("mechanism_of_action", ""),
        "replaced_atc": a.get("atc_classifications", ""),
        "candidate_atc": b.get("atc_classifications", ""),
        "moa_match_target": target_match,
        "moa_match_atc": atc_match,
        "moa_confidence": conf,
        "candidate_max_phase": b.get("max_phase", ""),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flow B: direct drug-level repurposing")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--cell-line", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--molecule-embeddings", default=DEFAULT_MOLECULE_EMBEDDINGS)
    p.add_argument("--fragment-embeddings", default=DEFAULT_FRAGMENT_EMBEDDINGS)
    p.add_argument("--fragment-map", default=DEFAULT_FRAGMENT_MAP)
    p.add_argument("--drug-smiles-map", default=DEFAULT_DRUG_SMILES_MAP)
    p.add_argument("--moa-cache", default="drug_moa_cache.csv")
    p.add_argument("--candidate-pool", choices=("vocab", "all"), default="vocab",
                   help="'vocab' restricts candidates to the encodable O'Neil drugs "
                        "the teacher was trained on -- an interpolation task. 'all' "
                        "opens the pool to every drug in dataset_EVERYTHING, which "
                        "LODO shows the teacher cannot rank (see --help notes).")
    p.add_argument("--novel-only", action="store_true",
                   help="Keep only substitutions whose resulting (cell line, pair) "
                        "combination was never measured, i.e. genuine predictions.")
    p.add_argument("--antagonism-threshold", type=float, default=-10.0,
                   help="Only pairs whose observed Loewe score is at or below "
                        "this value are treated as antagonistic.")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap the number of antagonistic pairs (most antagonistic first).")
    p.add_argument("--min-z", type=float, default=2.0,
                   help="Minimum z-score against the per-pair null to call a hit.")
    p.add_argument("--min-delta", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-fragments", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-all-scores", action="store_true",
                   help="Write the full (pair x side x candidate) score table. Large.")
    return p.parse_args()


def resolve_dataset(args, checkpoint) -> Path:
    if args.dataset:
        return Path(args.dataset)
    ds = checkpoint.get("dataset_path") or checkpoint.get("input")
    if ds:
        return Path(ds)
    raise ValueError("Dataset path not given and not recoverable from checkpoint.")


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device
    cell_name = args.cell_line

    log.info("Loading checkpoint -> %s", args.checkpoint)
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    model = build_model(checkpoint, device)
    model.eval()

    dataset_path = resolve_dataset(args, checkpoint)
    log.info("Loading dataset -> %s", dataset_path)
    dataset = load_pickle(dataset_path)
    meta = dataset["meta"]
    cell_omics = np.asarray(dataset["cell_omics"], dtype=np.float32)

    molecule_embeddings = load_embedding_pickle(Path(args.molecule_embeddings))
    fragment_embeddings = load_embedding_pickle(Path(args.fragment_embeddings))
    fragment_map = load_pickle(Path(args.fragment_map))
    drug_smiles_map = load_drug_smiles_map(Path(args.drug_smiles_map))

    # everything_drug_smiles_map.csv preserves the original drug casing, while
    # dataset meta uses its own; match case-insensitively.
    smiles_by_lower = {str(k).strip().lower(): v for k, v in drug_smiles_map.items()}

    # ---- antagonistic pairs in this cell line -----------------------------
    m = meta.copy()
    m["_row"] = np.arange(len(m))
    sel = m["cell_line_name"].astype(str).str.strip().str.lower() == cell_name.strip().lower()
    m = m[sel]
    if m.empty:
        raise SystemExit("Cell line %s not present in dataset." % cell_name)

    m["_a"] = m["drug_a"].astype(str).str.strip()
    m["_b"] = m["drug_b"].astype(str).str.strip()
    pairs = (m.groupby(["_a", "_b"], as_index=False)
               .agg(y_loewe=("y_loewe", "mean"), row=("_row", "first")))

    # Only 21 of the 38 O'Neil drugs carry a full feature set (graph embedding
    # plus a non-empty eMolFrag decomposition); the rest cannot be encoded, so
    # pairs involving them are dropped before any counting happens.
    def usable(name: str) -> bool:
        smi = smiles_by_lower.get(str(name).lower())
        if smi is None or molecule_embeddings.get(smi) is None:
            return False
        return _to_fragment_matrix(fragment_map.get(smi, []), fragment_embeddings) is not None

    encodable = {d for d in set(pairs._a) | set(pairs._b) if usable(d)}
    n_all = len(pairs)
    pairs = pairs[pairs._a.isin(encodable) & pairs._b.isin(encodable)]

    ant = pairs[pairs.y_loewe <= args.antagonism_threshold] \
              .sort_values("y_loewe").reset_index(drop=True)
    if args.max_pairs:
        ant = ant.head(args.max_pairs)

    log.info("Cell line %s: %d measured pairs, %d encodable, %d antagonistic "
             "(y_loewe <= %.1f); %d encodable drugs",
             cell_name, n_all, len(pairs), len(ant),
             args.antagonism_threshold, len(encodable))
    if ant.empty:
        raise SystemExit("No antagonistic pairs at this threshold.")

    # ---- candidate pool ---------------------------------------------------
    if args.candidate_pool == "vocab":
        vocab_map = {k: v for k, v in drug_smiles_map.items()
                     if str(k).strip() in encodable}
        pool = build_candidate_pool(vocab_map, molecule_embeddings,
                                    fragment_map, fragment_embeddings,
                                    args.max_fragments)
        log.info("Pool restricted to the teacher's training vocabulary "
                 "(interpolation regime)")
    else:
        pool = build_candidate_pool(drug_smiles_map, molecule_embeddings,
                                    fragment_map, fragment_embeddings,
                                    args.max_fragments)
        log.warning("Pool spans all of dataset_EVERYTHING. Leave-one-drug-out "
                    "shows the teacher cannot rank unseen drugs within a fixed "
                    "context (Spearman -0.03); treat output as exploratory.")
    if not pool:
        raise SystemExit("Empty candidate pool.")

    # Combinations already measured in this cell line. Where a recommendation
    # happens to have been measured we can check whether the improvement the
    # model promises is actually realised in the experimental data.
    observed_pair = {frozenset((a, b)): float(y)
                     for a, b, y in zip(pairs._a, pairs._b, pairs.y_loewe)}
    measured = set(observed_pair)

    moa_cache = load_moa_cache(args.moa_cache)

    total_scores = len(ant) * 2 * len(pool)
    log.info("Scoring %d pairs x 2 sides x %d candidates = %s forward passes",
             len(ant), len(pool), format(total_scores, ","))

    all_rows: List[dict] = []
    null_rows: List[dict] = []

    for pi, pr in ant.iterrows():
        da, db = pr["_a"], pr["_b"]
        smi_a = smiles_by_lower.get(da.lower())
        smi_b = smiles_by_lower.get(db.lower())
        if smi_a is None or smi_b is None:
            log.warning("  skip %s + %s (missing SMILES)", da, db)
            continue

        mol_a = molecule_embeddings.get(smi_a)
        mol_b = molecule_embeddings.get(smi_b)
        frag_a = _to_fragment_matrix(fragment_map.get(smi_a, []), fragment_embeddings)
        frag_b = _to_fragment_matrix(fragment_map.get(smi_b, []), fragment_embeddings)
        if mol_a is None or mol_b is None or frag_a is None or frag_b is None:
            log.warning("  skip %s + %s (missing features)", da, db)
            continue

        cell_vec = cell_omics[int(pr["row"])]
        s_orig = score_single(model, cell_vec, mol_a, mol_b, frag_a, frag_b, device)

        for side, replaced_name, partner_name, partner_mol, partner_frag in (
            ("a", da, db, mol_b, frag_b),
            ("b", db, da, mol_a, frag_a),
        ):
            preds = score_substitutions(model, cell_vec, partner_mol, partner_frag,
                                        pool, side, device, args.batch_size)
            deltas = preds - s_orig

            # per-pair null distribution over the whole candidate pool
            null_mean = float(deltas.mean())
            null_std = float(deltas.std()) or 1e-6
            order = deltas.argsort().argsort()
            pct = order / max(len(deltas) - 1, 1)

            null_rows.append({
                "cell_line": cell_name,
                "drug_a": da, "drug_b": db,
                "replaced_side": side,
                "replaced_drug": replaced_name,
                "partner_drug": partner_name,
                "observed_loewe": float(pr["y_loewe"]),
                "original_model_synergy": s_orig,
                "n_candidates": len(pool),
                "null_mean_delta": null_mean,
                "null_std_delta": null_std,
                "null_max_delta": float(deltas.max()),
            })

            for ci, cand in enumerate(pool):
                if cand.smiles == smi_a or cand.smiles == smi_b:
                    continue  # skip self and partner
                all_rows.append({
                    "cell_line": cell_name,
                    "drug_a": da, "drug_b": db,
                    "observed_loewe": float(pr["y_loewe"]),
                    "original_model_synergy": s_orig,
                    "replaced_side": side,
                    "replaced_drug": replaced_name,
                    "partner_drug": partner_name,
                    "candidate_drug": cand.name,
                    "candidate_smiles": cand.smiles,
                    "repurposed_synergy": float(preds[ci]),
                    "delta_synergy": float(deltas[ci]),
                    "z_vs_null": float((deltas[ci] - null_mean) / null_std),
                    "pct_rank_in_pool": float(pct[ci]),
                    "combination_is_novel":
                        frozenset((partner_name, cand.name)) not in measured,
                    "observed_loewe_of_recommendation":
                        observed_pair.get(frozenset((partner_name, cand.name)), np.nan),
                })

        if (pi + 1) % 5 == 0 or pi == len(ant) - 1:
            log.info("  scored pair %d/%d  (%s + %s)", pi + 1, len(ant), da, db)

    df = pd.DataFrame(all_rows)
    nulls = pd.DataFrame(null_rows)
    if df.empty:
        raise SystemExit("No pairs could be scored -- check SMILES and embedding coverage.")
    log.info("Scored %s substitutions", format(len(df), ","))

    # ---- MoA annotation ---------------------------------------------------
    log.info("Annotating mechanism of action ...")
    uniq = df[["replaced_drug", "candidate_drug"]].drop_duplicates()
    rel = {}
    for r in uniq.itertuples(index=False):
        rel[(r.replaced_drug, r.candidate_drug)] = moa_relation(
            r.replaced_drug, r.candidate_drug, moa_cache)
    for key in ("moa_confidence", "moa_match_target", "moa_match_atc",
                "candidate_moa", "replaced_moa", "candidate_max_phase"):
        df[key] = [rel[(a, b)][key] for a, b in
                   zip(df.replaced_drug, df.candidate_drug)]

    # ---- hits -------------------------------------------------------------
    hits = df[(df.z_vs_null >= args.min_z) & (df.delta_synergy >= args.min_delta)].copy()
    if args.novel_only:
        hits = hits[hits.combination_is_novel]
    conf_rank = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    hits["_c"] = hits.moa_confidence.map(conf_rank).fillna(3)
    hits = hits.sort_values(["_c", "z_vs_null"], ascending=[True, False]).drop(columns="_c")

    base = os.path.join(args.outdir, "flowB")
    nulls.to_csv("%s_null_stats_%s.csv" % (base, cell_name), index=False)
    hits.to_csv("%s_hits_%s.csv" % (base, cell_name), index=False)
    if args.save_all_scores:
        df.to_csv("%s_all_scores_%s.csv" % (base, cell_name), index=False)

    write_summary("%s_summary_%s.txt" % (base, cell_name), df, hits, nulls, args)

    log.info("Hits: %d  (MoA-high: %d)", len(hits),
             int((hits.moa_confidence == "high").sum()))
    log.info("Wrote results -> %s", args.outdir)


def write_summary(path, df, hits, nulls, args) -> None:
    L = []
    L.append("=" * 88)
    L.append("  FLOW B -- DRUG-LEVEL REPURPOSING SCORED DIRECTLY BY THE TEACHER")
    L.append("  cell line: %s" % args.cell_line)
    L.append("=" * 88)
    L.append("")
    L.append("Antagonistic pairs analysed : %d" % (len(nulls) // 2))
    L.append("Candidate pool              : %s (%d drugs)"
             % (args.candidate_pool, int(nulls.n_candidates.iloc[0])))
    L.append("Substitutions scored        : %s" % format(len(df), ","))
    if "combination_is_novel" in df.columns:
        L.append("Novel (unmeasured) combos   : %s of %s"
                 % (format(int(df.combination_is_novel.sum()), ","), format(len(df), ",")))
    L.append("")

    L.append("-" * 88)
    L.append("VALIDATION STATUS -- READ BEFORE USING THESE NUMBERS")
    L.append("-" * 88)
    if args.candidate_pool == "vocab":
        L.append("Candidates are drugs the teacher was trained on, so this is an")
        L.append("interpolation task. Combinations flagged as novel were never measured")
        L.append("in this cell line and are genuine predictions; the rest have known")
        L.append("answers and are included only as internal controls.")
    else:
        L.append("Candidates include drugs absent from the teacher's training set.")
        L.append("Leave-one-drug-out analysis shows that within a fixed (cell line,")
        L.append("partner) context the teacher ranks unseen drugs no better than chance:")
        L.append("  Spearman(predicted, observed) = -0.03   (in-sample: +0.87)")
        L.append("  top-1 is the true best        =  4.8%   (chance 6.0%)")
        L.append("  mean regret                   = 17.6    (random picker 17.7)")
        L.append("Its apparent skill comes entirely from cell-line and partner-drug")
        L.append("effects: removing those from the pooled LODO correlation takes it from")
        L.append("+0.288 to -0.016. Rankings below are therefore exploratory only.")
    L.append("")
    L.append("Gates:  z >= %.1f against the per-pair null AND delta >= %.1f"
             % (args.min_z, args.min_delta))
    L.append("Hits passing gates          : %d" % len(hits))
    for c in ("high", "medium", "low", "unknown"):
        L.append("    MoA %-8s : %d" % (c, int((hits.moa_confidence == c).sum())))
    L.append("")

    L.append("-" * 88)
    L.append("REGRESSION-TO-THE-MEAN CONTROL")
    L.append("-" * 88)
    L.append("The mean delta over the whole candidate pool is the improvement obtained")
    L.append("by an arbitrary substitution. It is the baseline a real candidate must beat.")
    L.append("")
    L.append("  mean null delta over all pair-sides : %+.2f" % nulls.null_mean_delta.mean())
    L.append("  sd   null delta                     :  %.2f" % nulls.null_std_delta.mean())
    if len(nulls) > 2:
        r = np.corrcoef(nulls.original_model_synergy, nulls.null_mean_delta)[0, 1]
        L.append("  corr(original synergy, null mean)   : %+.3f" % r)
        L.append("  -> a strongly negative correlation confirms that more antagonistic")
        L.append("     pairs show larger apparent gains for ANY substitution, which is")
        L.append("     why the z-score rather than the raw delta is the reported statistic.")
    L.append("")

    col = "observed_loewe_of_recommendation"
    if col in hits.columns:
        chk = hits.dropna(subset=[col])
        if len(chk):
            realised = chk[col] > chk.observed_loewe
            L.append("-" * 88)
            L.append("GROUND-TRUTH CHECK ON RECOMMENDATIONS THAT WERE ACTUALLY MEASURED")
            L.append("-" * 88)
            L.append("Some recommended combinations happen to exist in O'Neil for this cell")
            L.append("line. For those the promised improvement can be checked directly against")
            L.append("measurement rather than against another model prediction.")
            L.append("")
            L.append("  recommendations with a measured outcome : %d of %d hits"
                     % (len(chk), len(hits)))
            L.append("  improvement realised experimentally     : %d (%.1f%%)"
                     % (int(realised.sum()), 100.0 * realised.mean()))
            L.append("  mean observed synergy, original pair    : %+.2f" % chk.observed_loewe.mean())
            L.append("  mean observed synergy, recommended pair : %+.2f" % chk[col].mean())
            L.append("  mean realised gain                      : %+.2f"
                     % (chk[col] - chk.observed_loewe).mean())
            L.append("  mean predicted gain                     : %+.2f" % chk.delta_synergy.mean())
            if len(chk) > 2:
                rr = np.corrcoef(chk.delta_synergy, chk[col] - chk.observed_loewe)[0, 1]
                L.append("  corr(predicted gain, realised gain)     : %+.3f" % rr)
            L.append("")

            # Ground-truth null: the gain an arbitrary in-vocabulary substitution
            # would have delivered for the same pairs, measured experimentally.
            # No model output enters this baseline.
            allm = df.dropna(subset=[col])
            if len(allm):
                allm = allm.assign(gain=allm[col] - allm.observed_loewe)
                keys = ["drug_a", "drug_b", "replaced_side"]
                sel = chk[keys].drop_duplicates()
                base = allm.merge(sel, on=keys).gain
                hit_gain = (chk[col] - chk.observed_loewe)
                L.append("  BASELINE -- random measured substitution into the same pairs")
                L.append("    mean gain of an arbitrary substitution: %+.2f  (n=%d)"
                         % (base.mean(), len(base)))
                L.append("    mean gain of the selected hits        : %+.2f  (n=%d)"
                         % (hit_gain.mean(), len(hit_gain)))
                L.append("    advantage over arbitrary substitution : %+.2f"
                         % (hit_gain.mean() - base.mean()))
                L.append("    arbitrary substitutions that improve  : %.1f%%"
                         % (100.0 * (base > 0).mean()))
                L.append("  Most of the apparent gain is available from any substitution;")
                L.append("  the advantage line is the part attributable to the model.")
            L.append("")

    L.append("-" * 88)
    L.append("TOP HITS  (MoA-high first, then by z-score)")
    L.append("-" * 88)
    if hits.empty:
        L.append("  none passed the gates")
    else:
        for i, h in enumerate(hits.head(40).itertuples(index=False), 1):
            L.append("%3d. %s + %s   [%s]" % (i, h.drug_a, h.drug_b, h.moa_confidence))
            L.append("     replace %-28s -> %s" % (h.replaced_drug, h.candidate_drug))
            L.append("     synergy %+.2f -> %+.2f   (delta %+.2f, z=%.2f, top %.2f%% of pool)"
                     % (h.original_model_synergy, h.repurposed_synergy,
                        h.delta_synergy, h.z_vs_null,
                        100.0 * (1.0 - h.pct_rank_in_pool)))
            if h.candidate_moa:
                L.append("     candidate MoA: %s" % str(h.candidate_moa)[:70])
            L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
