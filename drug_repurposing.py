"""
drug_repurposing.py
====================
Drug repurposing extension for the eComb2 bioisostere pipeline.

Given the rescored candidate output from rescore_reconstructed_candidates.py,
this script identifies existing drugs in dataset_EVERYTHING that already
contain the optimal replacement fragment — meaning they can be used directly
in the drug combination *without* synthesising a new molecule.

For each hit it reports:
  • original_synergy       — teacher model score for the original pair (Loewe)
  • estimated_new_synergy  — teacher model score after fragment swap
  • delta_synergy          — improvement = new − original

Matching strategy (three tiers, reported separately):
  1. EXACT    — candidate_fragment_smiles is an exact canonical SMILES match
                to a known BRICS fragment of an existing drug.
  2. SUBSTR   — candidate fragment is a substructure of an existing drug
                (only fragments with ≥ --min-fragment-atoms heavy atoms,
                 covering ≥ --min-fragment-coverage of the drug's atoms).
  3. TANIMOTO — Morgan fingerprint Tanimoto similarity ≥ --tanimoto-threshold.

Output
------
  repurposing_candidates_<cell>.csv   — full hit table
  repurposing_summary_<cell>.txt      — human-readable ranked summary
  repurposing_lollipop_<cell>.svg     — before/after synergy lollipop chart
  repurposing_scatter_<cell>.svg      — original vs repurposed scatter plot
  repurposing_topbars_<cell>.svg      — delta improvement bar chart

Usage
-----
  python drug_repurposing.py \\
      --rescored   rescored_reconstructed_candidates.csv \\
      --frag-map   fragment_ae/smiles_to_fragments.pkl \\
      --drug-map   everything_drug_smiles_map.csv \\
      --outdir     repurposing/ \\
      --cell-line  A2058 \\
      [--exact-only] \\
      [--tanimoto-threshold 0.5] \\
      [--top-n 20] \\
      [--min-delta 0.0]
"""

import argparse
import csv
import logging
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    log.warning("RDKit not available — substructure and Tanimoto matching disabled.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MPL_OK = True
except ImportError:
    MPL_OK = False
    log.warning("matplotlib not available — plots will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Cell-line colour palette (matches thesis figures)
# ─────────────────────────────────────────────────────────────────────────────
CELL_COLOURS = {
    "A2058":  {"fill": "#FFAAAA", "edge": "#CC0000", "dark": "#CC0000"},
    "HCT116": {"fill": "#AAFFAA", "edge": "#007700", "dark": "#007700"},
    "default":{"fill": "#AACCFF", "edge": "#0044AA", "dark": "#0044AA"},
}

def cell_colour(cell_line: str) -> dict:
    return CELL_COLOURS.get(cell_line, CELL_COLOURS["default"])


# ─────────────────────────────────────────────────────────────────────────────
# RDKit helpers
# ─────────────────────────────────────────────────────────────────────────────

def canonical(smi: str) -> Optional[str]:
    if not RDKIT_OK or not smi:
        return smi
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def morgan_fp(smi: str):
    if not RDKIT_OK or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def has_substructure(drug_smi: str, frag_smi: str,
                     min_coverage: float = 0.20) -> bool:
    """
    True if frag_smi is a substructure of drug_smi AND the fragment covers
    at least min_coverage of the drug's heavy atoms.
    """
    if not RDKIT_OK:
        return False
    drug_mol = Chem.MolFromSmiles(drug_smi)
    frag_mol = Chem.MolFromSmiles(frag_smi)
    if drug_mol is None or frag_mol is None:
        return False
    if not drug_mol.HasSubstructMatch(frag_mol):
        return False
    frag_heavy = frag_mol.GetNumHeavyAtoms()
    drug_heavy = drug_mol.GetNumHeavyAtoms()
    if drug_heavy == 0:
        return False
    return (frag_heavy / drug_heavy) >= min_coverage


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_fragment_map(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    log.info("Loaded fragment map: %d drugs", len(raw))
    return raw


def build_inverted_index(frag_map: dict) -> dict:
    inv = defaultdict(list)
    for drug_smi, frags in frag_map.items():
        for frag in frags:
            canon = canonical(frag) or frag
            inv[canon].append(drug_smi)
    log.info("Inverted fragment index: %d unique fragments", len(inv))
    return dict(inv)


def load_drug_name_map(csv_path: str) -> Tuple[dict, dict]:
    smiles_to_name, name_to_smiles = {}, {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            smi  = row.get("smiles", "").strip()
            name = row.get("drug", "").strip()
            if smi and name:
                smiles_to_name[smi]  = name
                name_to_smiles[name] = smi
    log.info("Drug name map: %d entries", len(smiles_to_name))
    return smiles_to_name, name_to_smiles


def load_rescored(csv_path: str, min_delta: float) -> List[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                delta = float(row["rescored_delta_synergy_vs_model"])
            except (ValueError, KeyError):
                continue
            if delta >= min_delta:
                row["_delta"] = delta
                try:
                    row["_orig"]    = float(row.get("teacher_original_synergy", "nan"))
                    row["_new"]     = float(row.get("replaced_predicted_synergy", "nan"))
                    row["_obs"]     = float(row.get("original_observed_synergy", "nan"))
                except ValueError:
                    row["_orig"] = row["_new"] = row["_obs"] = float("nan")
                rows.append(row)
    rows.sort(key=lambda r: -r["_delta"])
    log.info("Loaded %d rescored candidates (min_delta=%.1f)", len(rows), min_delta)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────

def match_candidates(
    rescored_rows: List[dict],
    frag_map: dict,
    inv_index: dict,
    smiles_to_name: dict,
    tanimoto_threshold: float,
    top_n: int,
    min_fragment_coverage: float = 0.20,
    min_fragment_atoms: int = 8,
    exact_only: bool = False,
) -> List[dict]:
    """
    For each rescored candidate, find existing drugs that contain the
    candidate fragment (exact, substructure, or Tanimoto).
    """
    hits = []
    seen = set()
    drug_fp_cache: Dict[str, object] = {}

    # expose to inner scope
    args_coverage       = min_fragment_coverage
    args_min_frag_atoms = min_fragment_atoms
    args_exact_only     = exact_only

    for row in rescored_rows:
        cand_smi = row.get("candidate_fragment_smiles", "").strip()
        if not cand_smi:
            continue

        canon_cand = canonical(cand_smi) or cand_smi
        cand_fp    = morgan_fp(canon_cand)

        orig_syn  = row.get("_orig", float("nan"))
        new_syn   = row.get("_new",  float("nan"))
        obs_syn   = row.get("_obs",  float("nan"))

        matched_drugs: List[Tuple[str, str, float]] = []

        # ── Tier 1: exact fragment match ──────────────────────────────────────
        if canon_cand in inv_index:
            for drug_smi in inv_index[canon_cand]:
                matched_drugs.append((drug_smi, "EXACT", 1.0))

        # ── Tier 2: substructure match ────────────────────────────────────────
        exact_smis = {d[0] for d in matched_drugs}
        if RDKIT_OK and not args_exact_only:
            frag_mol_size = Chem.MolFromSmiles(canon_cand)
            frag_heavy = frag_mol_size.GetNumHeavyAtoms() if frag_mol_size else 0
            if frag_heavy >= args_min_frag_atoms:
                for drug_smi in frag_map:
                    if drug_smi in exact_smis:
                        continue
                    if has_substructure(drug_smi, canon_cand,
                                        min_coverage=args_coverage):
                        matched_drugs.append((drug_smi, "SUBSTR", 1.0))

        # ── Tier 3: Tanimoto ─────────────────────────────────────────────────
        already_matched = {d[0] for d in matched_drugs}
        if RDKIT_OK and cand_fp is not None and not args_exact_only:
            for drug_smi in frag_map:
                if drug_smi in already_matched:
                    continue
                if drug_smi not in drug_fp_cache:
                    drug_fp_cache[drug_smi] = morgan_fp(drug_smi)
                sim = tanimoto(cand_fp, drug_fp_cache[drug_smi])
                if sim >= tanimoto_threshold:
                    matched_drugs.append((drug_smi, "TANIMOTO", round(sim, 4)))

        # ── Emit one hit row per matched drug ─────────────────────────────────
        for drug_smi, tier, sim in matched_drugs:
            drug_name = smiles_to_name.get(drug_smi, drug_smi[:40] + "…")

            replace_drug_pos = row.get("replace_drug", "")
            replaced_name = (
                row.get("drug_a", "") if replace_drug_pos == "a"
                else row.get("drug_b", "")
            )
            partner_name = (
                row.get("drug_b", "") if replace_drug_pos == "a"
                else row.get("drug_a", "")
            )

            key = (
                row.get("cell_line_name", ""),
                row.get("drug_a", ""),
                row.get("drug_b", ""),
                replace_drug_pos,
                row.get("source_fragment_smiles", ""),
                canon_cand,
                drug_name,
                tier,
            )
            if key in seen:
                continue
            seen.add(key)

            # Don't suggest a drug to replace itself
            if drug_name.lower() in (
                row.get("drug_a", "").lower(),
                row.get("drug_b", "").lower(),
            ):
                continue

            hits.append({
                # --- identification ---
                "cell_line":               row.get("cell_line_name", ""),
                "original_drug_a":         row.get("drug_a", ""),
                "original_drug_b":         row.get("drug_b", ""),
                "replaced_drug":           replaced_name,
                "partner_drug":            partner_name,
                "repurpose_drug_name":     drug_name,
                "repurpose_drug_smiles":   drug_smi,
                # --- fragment info ---
                "source_fragment":         row.get("source_fragment_smiles", ""),
                "candidate_fragment":      canon_cand,
                # --- synergy scores (the key numbers the user asked for) ---
                "original_observed_synergy":     round(obs_syn, 3) if obs_syn == obs_syn else "",
                "original_model_synergy":        round(orig_syn, 3) if orig_syn == orig_syn else "",
                "estimated_repurposed_synergy":  round(new_syn, 3)  if new_syn  == new_syn  else "",
                "delta_synergy":                 round(row["_delta"], 3),
                # --- match quality ---
                "selection":               row.get("selection", ""),
                "match_tier":              tier,
                "tanimoto_similarity":     sim,
                # --- interpretive label ---
                "recommendation": (
                    f"Replace {replaced_name} with {drug_name} "
                    f"(partner: {partner_name}); "
                    f"synergy {orig_syn:+.1f} → {new_syn:+.1f} Loewe"
                    if orig_syn == orig_syn and new_syn == new_syn
                    else f"Replace {replaced_name} with {drug_name} (partner: {partner_name})"
                ),
            })

    tier_rank = {"EXACT": 0, "SUBSTR": 1, "TANIMOTO": 2}
    hits.sort(key=lambda h: (-h["delta_synergy"], tier_rank[h["match_tier"]]))
    return hits[:top_n] if top_n > 0 else hits


# ─────────────────────────────────────────────────────────────────────────────
# MoA cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_moa_cache(csv_path: str) -> Dict[str, dict]:
    """
    Load drug_moa_cache.csv produced by build_moa_cache.py.
    Returns {drug_name_lower: row_dict}.
    """
    cache: Dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cache[row["drug_name"].lower()] = row
    log.info("MoA cache loaded: %d entries", len(cache))
    return cache


def _moa_lookup(name: str, cache: Dict[str, dict]) -> dict:
    """Return a cache row for `name` (case-insensitive), or empty dict."""
    return cache.get(name.lower(), {})


def _shared_atc_l1(row_a: dict, row_b: dict) -> bool:
    """True if both drugs have at least one ATC L1 code in common."""
    def atc_l1_set(row: dict) -> Set[str]:
        codes = row.get("atc_classifications", "")
        return {c.strip()[0].upper() for c in codes.split(",") if c.strip()}
    a = atc_l1_set(row_a)
    b = atc_l1_set(row_b)
    return bool(a and b and a & b)


def _shared_target(row_a: dict, row_b: dict) -> bool:
    """True if both drugs share at least one ChEMBL target ID."""
    def target_set(row: dict) -> Set[str]:
        t = row.get("target_chembl_ids", "")
        return {x.strip() for x in t.split(";") if x.strip()}
    a = target_set(row_a)
    b = target_set(row_b)
    return bool(a and b and a & b)


def annotate_moa(hits: List[dict], cache: Dict[str, dict]) -> List[dict]:
    """
    Add MoA columns to each hit dict in-place.  Fields added:

      replaced_moa          — MoA text of the drug being replaced
      replaced_atc          — ATC code(s) of the replaced drug
      repurpose_moa         — MoA text of the repurposing candidate
      repurpose_atc         — ATC code(s) of the repurposing candidate
      moa_match_atc         — True if both share the same ATC L1 letter
      moa_match_target      — True if both share at least one ChEMBL target
      moa_match             — True if either of the above is True
      moa_confidence        — 'high' (target match), 'medium' (ATC match),
                              'low' (no match found), 'unknown' (no cache data)
    """
    for h in hits:
        r_row  = _moa_lookup(h["replaced_drug"],        cache)
        rp_row = _moa_lookup(h["repurpose_drug_name"],  cache)

        h["replaced_moa"]   = r_row.get("mechanism_of_action", "")
        h["replaced_atc"]   = r_row.get("atc_classifications", "")
        h["repurpose_moa"]  = rp_row.get("mechanism_of_action", "")
        h["repurpose_atc"]  = rp_row.get("atc_classifications", "")

        atc_match    = _shared_atc_l1(r_row, rp_row)
        target_match = _shared_target(r_row, rp_row)

        h["moa_match_atc"]    = atc_match
        h["moa_match_target"] = target_match
        h["moa_match"]        = atc_match or target_match

        if not r_row or not rp_row:
            h["moa_confidence"] = "unknown"
        elif target_match:
            h["moa_confidence"] = "high"
        elif atc_match:
            h["moa_confidence"] = "medium"
        else:
            h["moa_confidence"] = "low"

    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "cell_line",
    "original_drug_a",
    "original_drug_b",
    "replaced_drug",
    "partner_drug",
    "repurpose_drug_name",
    "source_fragment",
    "candidate_fragment",
    "original_observed_synergy",
    "original_model_synergy",
    "estimated_repurposed_synergy",
    "delta_synergy",
    "selection",
    "match_tier",
    "tanimoto_similarity",
    # MoA columns (populated when --moa-cache is provided)
    "replaced_moa",
    "replaced_atc",
    "repurpose_moa",
    "repurpose_atc",
    "moa_match_atc",
    "moa_match_target",
    "moa_match",
    "moa_confidence",
    "recommendation",
    "repurpose_drug_smiles",
]


def write_csv(path: str, hits: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(hits)
    log.info("Wrote %d hits → %s", len(hits), path)


def write_summary(path: str, hits: List[dict], args: argparse.Namespace) -> None:
    # Sort: MoA confidence first (high > medium > low > unknown), then ΔS desc
    _conf_rank = {"high": 0, "medium": 1, "low": 2, "unknown": 3, "": 4}
    sorted_hits = sorted(
        hits,
        key=lambda h: (
            _conf_rank.get(h.get("moa_confidence", ""), 4),
            -float(h["delta_synergy"]),
        ),
    )

    tier_counts: Dict[str, int] = defaultdict(int)
    for h in hits:
        tier_counts[h["match_tier"]] += 1

    conf_counts: Dict[str, int] = defaultdict(int)
    for h in hits:
        conf_counts[h.get("moa_confidence", "")] += 1

    drug_freq: Dict[str, int] = defaultdict(int)
    for h in hits:
        drug_freq[h["repurpose_drug_name"]] += 1

    moa_enabled = any(h.get("moa_confidence") for h in hits)

    lines = [
        "=" * 80,
        "  eComb2  Drug Repurposing Report",
        "=" * 80,
        "  Cell line      : %s" % (args.cell_line or "all"),
        "  Min ΔS filter  : %+.1f Loewe units" % args.min_delta,
        "  Match tiers    : %s" % ("EXACT only" if args.exact_only
                                    else "EXACT + SUBSTR + TANIMOTO"),
        "  MoA cache      : %s" % (args.moa_cache or "not used"),
        "",
        "  Total repurposing hits : %d" % len(hits),
        "    EXACT     : %d" % tier_counts["EXACT"],
        "    SUBSTR    : %d" % tier_counts["SUBSTR"],
        "    TANIMOTO  : %d" % tier_counts["TANIMOTO"],
    ]

    if moa_enabled:
        lines += [
            "",
            "  MoA confidence breakdown:",
            "    ★★★ high    (same ChEMBL target)  : %d" % conf_counts.get("high", 0),
            "    ★★  medium  (same ATC L1 class)   : %d" % conf_counts.get("medium", 0),
            "    ★   low     (different class)      : %d" % conf_counts.get("low", 0),
            "        unknown (no cache data)        : %d" % conf_counts.get("unknown", 0),
        ]

    lines += [
        "",
        "─" * 80,
        "  Most frequently matched repurposing drugs:",
        "─" * 80,
    ]
    for drug, cnt in sorted(drug_freq.items(), key=lambda x: -x[1])[:15]:
        lines.append("    %3d×  %s" % (cnt, drug))

    # ── Render hits grouped by MoA confidence ────────────────────────────────
    def _section(label: str, group_hits: List[dict], start_idx: int) -> Tuple[List[str], int]:
        if not group_hits:
            return [], start_idx
        sec = [
            "",
            "─" * 80,
            "  %s" % label,
            "─" * 80,
        ]
        i = start_idx
        for h in group_hits:
            orig = h["original_model_synergy"]
            new  = h["estimated_repurposed_synergy"]
            try:
                score_str = "%+6.1f → %+6.1f" % (float(orig), float(new))
            except (ValueError, TypeError):
                score_str = "   n/a →    n/a"
            pair = "%s + %s" % (h["original_drug_a"], h["original_drug_b"])
            sec.append(
                "  %3d  [%5s]  ΔS=%+7.2f   %s   %s → %s"
                % (i, h["match_tier"], float(h["delta_synergy"]),
                   score_str, pair, h["repurpose_drug_name"])
            )
            sec.append(
                "       fragment swap : %s → %s"
                % (h["source_fragment"], h["candidate_fragment"])
            )
            if h.get("repurpose_moa"):
                sec.append("       repurpose MoA: %s" % h["repurpose_moa"])
            if h.get("replaced_moa"):
                sec.append("       replaced  MoA: %s" % h["replaced_moa"])
            if h.get("repurpose_atc"):
                sec.append("       repurpose ATC: %s" % h["repurpose_atc"])
            sec.append("")
            i += 1
        return sec, i

    if moa_enabled:
        by_conf: Dict[str, List[dict]] = defaultdict(list)
        for h in sorted_hits:
            by_conf[h.get("moa_confidence", "unknown")].append(h)

        idx = 1
        for conf, label in [
            ("high",    "★★★ HIGH confidence — same ChEMBL target as replaced drug"),
            ("medium",  "★★  MEDIUM confidence — same ATC therapeutic class"),
            ("low",     "★   LOW confidence — different therapeutic class (fragment match only)"),
            ("unknown", "    UNKNOWN confidence — MoA data not available"),
        ]:
            sec, idx = _section(label, by_conf.get(conf, []), idx)
            lines.extend(sec)
    else:
        # No MoA cache — flat list sorted by delta
        flat_sec, _ = _section(
            "All repurposing candidates (sorted by ΔS)",
            sorted_hits[:50], 1
        )
        lines.extend(flat_sec)

    lines.append("=" * 80)
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(text)
    log.info("Summary → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _shorten(name: str, max_len: int = 20) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


MOA_CONF_STYLE = {
    "high":    {"marker": "*", "size": 160, "label": "★★★ High (same target)"},
    "medium":  {"marker": "D", "size": 100, "label": "★★  Medium (same ATC)"},
    "low":     {"marker": "o", "size":  80, "label": "★   Low (diff. class)"},
    "unknown": {"marker": "s", "size":  70, "label": "    Unknown"},
    "":        {"marker": "o", "size":  80, "label": ""},
}


def _sorted_for_plot(hits: List[dict], top_n: int) -> List[dict]:
    """Return hits sorted: high-confidence first, then by delta, capped at top_n."""
    _rank = {"high": 0, "medium": 1, "low": 2, "unknown": 3, "": 4}
    return sorted(
        [h for h in hits
         if h.get("original_model_synergy", "") != ""
         and h.get("estimated_repurposed_synergy", "") != ""],
        key=lambda h: (
            _rank.get(h.get("moa_confidence", ""), 4),
            -float(h["delta_synergy"]),
        ),
    )[:top_n]


def plot_lollipop(hits: List[dict], outpath: str, cell_line: str,
                  top_n: int = 20) -> None:
    """
    Lollipop chart: each row = one repurposing hit.
    Left dot  = original synergy score (grey).
    Right dot = estimated repurposed synergy, coloured by MoA confidence.
    Horizontal line connects them; arrow shows direction of change.
    """
    if not MPL_OK:
        return

    subset = _sorted_for_plot(hits, top_n)
    if not subset:
        log.warning("No hits with synergy scores for lollipop plot.")
        return

    cc  = cell_colour(cell_line)
    n   = len(subset)
    fig, ax = plt.subplots(figsize=(13, max(6, n * 0.55)))

    y_positions = list(range(n))
    orig_vals   = [float(h["original_model_synergy"])        for h in subset]
    new_vals    = [float(h["estimated_repurposed_synergy"])   for h in subset]
    deltas      = [float(h["delta_synergy"])                  for h in subset]

    # Draw connecting lines
    for y, o, nv in zip(y_positions, orig_vals, new_vals):
        ax.hlines(y, min(o, nv), max(o, nv), color="#AAAAAA", linewidth=1.8, zorder=1)

    # Arrows to show direction
    for y, o, nv in zip(y_positions, orig_vals, new_vals):
        mid = (o + nv) / 2.0
        ax.annotate(
            "",
            xy=(nv, y), xytext=(mid, y),
            arrowprops=dict(arrowstyle="->", color=cc["dark"], lw=1.5),
            zorder=2,
        )

    # Scatter: original (grey circles)
    ax.scatter(orig_vals, y_positions,
               color="#CCCCCC", edgecolors="#555555", s=80, zorder=3,
               label="Original synergy", marker="o")

    # Repurposed dots — colour and shape by MoA confidence
    _moa_colour = {"high": "#007700", "medium": "#0044CC",
                   "low": cc["fill"], "unknown": "#AAAAAA", "": cc["fill"]}
    _moa_edge   = {"high": "#003300", "medium": "#001166",
                   "low": cc["edge"], "unknown": "#555555", "": cc["edge"]}
    _added_labels: Set[str] = set()
    for y, nv, h in zip(y_positions, new_vals, subset):
        conf  = h.get("moa_confidence", "")
        style = MOA_CONF_STYLE.get(conf, MOA_CONF_STYLE[""])
        lbl   = style["label"] if style["label"] not in _added_labels else "_"
        _added_labels.add(style["label"])
        ax.scatter([nv], [y],
                   color=_moa_colour[conf], edgecolors=_moa_edge[conf],
                   s=style["size"], zorder=4, marker=style["marker"],
                   label=lbl)

    # Delta annotation on the right
    xmax = ax.get_xlim()[1] if ax.get_xlim()[1] != 0 else max(new_vals) + 5
    for y, d in zip(y_positions, deltas):
        ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] < 0 else max(new_vals) + 1,
                y, f"ΔS={d:+.1f}", va="center", ha="left",
                fontsize=8, color="#000000")

    # Y-axis labels: "replaced_drug → repurpose_drug"
    ylabels = [
        f"{_shorten(h['replaced_drug'],18)} → {_shorten(h['repurpose_drug_name'],18)}\n"
        f"  ({_shorten(h['partner_drug'],16)} combo)"
        for h in subset
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, fontsize=8.5, color="#000000")

    # Zero line (reference: 0 = additive)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Additive (Loewe = 0)")

    ax.set_xlabel("Loewe Synergy Score  (more negative = more antagonistic)",
                  fontsize=11, color="#000000", fontweight="bold")
    ax.set_title(
        f"Drug Repurposing — Before vs. After  [{cell_line}]\n"
        f"Original synergy → Estimated synergy after replacing with repurposed drug",
        fontsize=12, color="#000000", fontweight="bold", pad=10,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.tick_params(colors="#000000")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info("Lollipop plot → %s", outpath)


def plot_scatter(hits: List[dict], outpath: str, cell_line: str) -> None:
    """
    Scatter: original synergy (x) vs estimated repurposed synergy (y).
    Points above y=x line → improvement. Diagonal = no change.
    """
    if not MPL_OK:
        return

    valid = [h for h in hits
             if h["original_model_synergy"] != "" and h["estimated_repurposed_synergy"] != ""]
    if not valid:
        return

    cc = cell_colour(cell_line)
    orig = np.array([float(h["original_model_synergy"])       for h in valid])
    new  = np.array([float(h["estimated_repurposed_synergy"]) for h in valid])

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(orig, new, color=cc["fill"], edgecolors=cc["edge"],
               s=70, alpha=0.85, zorder=3)

    # Diagonal y = x (no improvement)
    lim = [min(orig.min(), new.min()) - 5, max(orig.max(), new.max()) + 5]
    ax.plot(lim, lim, "--", color="#888888", linewidth=1.2, label="No change (y = x)")

    # Shade improvement region
    ax.fill_between(lim, lim, [lim[1], lim[1]],
                    color=cc["fill"], alpha=0.08, label="Improvement region")

    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax.set_xlabel("Original Synergy Score (Loewe)", fontsize=11,
                  fontweight="bold", color="#000000")
    ax.set_ylabel("Estimated Repurposed Synergy (Loewe)", fontsize=11,
                  fontweight="bold", color="#000000")
    ax.set_title(
        f"Original vs. Repurposed Synergy  [{cell_line}]\n"
        f"(n = {len(valid)} hits;  points above diagonal = improved)",
        fontsize=11, color="#000000", fontweight="bold",
    )
    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax.axvline(0, color="#AAAAAA", linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.tick_params(colors="#000000")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate a few top hits
    top5 = sorted(valid, key=lambda h: -float(h["delta_synergy"]))[:5]
    for h in top5:
        ox, ny = float(h["original_model_synergy"]), float(h["estimated_repurposed_synergy"])
        label  = _shorten(h["repurpose_drug_name"], 14)
        ax.annotate(label, (ox, ny), textcoords="offset points",
                    xytext=(4, 3), fontsize=7, color=cc["dark"])

    fig.tight_layout()
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info("Scatter plot → %s", outpath)


def plot_delta_bars(hits: List[dict], outpath: str, cell_line: str,
                    top_n: int = 20) -> None:
    """
    Horizontal bar chart of ΔS for the top-N hits, sorted: high-conf first,
    then by delta. Bars are colour-coded by MoA confidence.
    """
    if not MPL_OK:
        return

    subset = _sorted_for_plot(hits, top_n)
    if not subset:
        return

    cc  = cell_colour(cell_line)
    n   = len(subset)
    fig, ax = plt.subplots(figsize=(12, max(5, n * 0.52)))

    labels = [
        "%s → %s\n  + %s" % (
            _shorten(h["replaced_drug"], 15),
            _shorten(h["repurpose_drug_name"], 15),
            _shorten(h["partner_drug"], 14),
        )
        for h in subset
    ]
    deltas = [float(h["delta_synergy"]) for h in subset]
    y      = list(range(n))

    # Bar colours by MoA confidence
    _bar_fill = {"high": "#CCFFCC", "medium": "#CCE0FF",
                 "low": cc["fill"], "unknown": "#EEEEEE", "": cc["fill"]}
    _bar_edge = {"high": "#007700", "medium": "#0044CC",
                 "low": cc["edge"], "unknown": "#999999", "": cc["edge"]}

    bar_fills = [_bar_fill.get(h.get("moa_confidence", ""), cc["fill"]) for h in subset]
    bar_edges = [_bar_edge.get(h.get("moa_confidence", ""), cc["edge"]) for h in subset]

    bars = ax.barh(y, deltas, color=bar_fills, edgecolor=bar_edges,
                   linewidth=1.2, height=0.65)

    # Annotate each bar with orig → new
    for i, (bar, h) in enumerate(zip(bars, subset)):
        orig = h["original_model_synergy"]
        new  = h["estimated_repurposed_synergy"]
        try:
            label = f"{float(orig):+.1f} → {float(new):+.1f}"
        except (ValueError, TypeError):
            label = ""
        x_right = bar.get_width()
        ax.text(x_right + 0.5, i, label, va="center", ha="left",
                fontsize=8, color="#000000", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color="#000000")
    ax.set_xlabel("ΔS  (Estimated synergy improvement, Loewe units)",
                  fontsize=11, fontweight="bold", color="#000000")
    ax.set_title(
        "Top Repurposing Candidates — Synergy Improvement  [%s]\n"
        "Sorted: high MoA confidence first, then by ΔS. "
        "Bar colour = MoA confidence." % cell_line,
        fontsize=11, color="#000000", fontweight="bold", pad=10,
    )
    ax.axvline(0, color="#888888", linewidth=1.0)
    ax.tick_params(colors="#000000")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # Legend for MoA confidence colours
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(facecolor="#CCFFCC", edgecolor="#007700", label="★★★ High (same target)"),
        mpatches.Patch(facecolor="#CCE0FF", edgecolor="#0044CC", label="★★  Medium (same ATC)"),
        mpatches.Patch(facecolor=cc["fill"], edgecolor=cc["edge"], label="★   Low / Unknown"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.85)

    # Give extra x-space for annotations
    x_max = max(deltas) if deltas else 1
    ax.set_xlim(0, x_max * 1.42)

    fig.tight_layout()
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info("Delta bar chart → %s", outpath)


def plot_synergy_panels(hits: List[dict], outpath: str, cell_line: str,
                        top_n: int = 15) -> None:
    """
    Two-panel figure: lollipop (high-conf first) + scatter.
    Combined into one file for easy embedding in thesis.
    """
    if not MPL_OK:
        return

    subset = _sorted_for_plot(hits, top_n)
    if not subset:
        return

    cc   = cell_colour(cell_line)
    orig = [float(h["original_model_synergy"])       for h in subset]
    new  = [float(h["estimated_repurposed_synergy"]) for h in subset]
    delt = [float(h["delta_synergy"])                for h in subset]
    n    = len(subset)

    fig, (ax_lol, ax_sc) = plt.subplots(1, 2, figsize=(16, max(6, n * 0.5)),
                                         gridspec_kw={"width_ratios": [2, 1]})

    # ── Lollipop ─────────────────────────────────────────────────────────────
    y = list(range(n))
    for yi, o, nv in zip(y, orig, new):
        ax_lol.hlines(yi, min(o, nv), max(o, nv), color="#CCCCCC", linewidth=2, zorder=1)
    for yi, o, nv in zip(y, orig, new):
        mid = (o + nv) / 2.0
        ax_lol.annotate("", xy=(nv, yi), xytext=(mid, yi),
                         arrowprops=dict(arrowstyle="->", color=cc["dark"], lw=1.5),
                         zorder=2)
    ax_lol.scatter(orig, y, color="#DDDDDD", edgecolors="#666666", s=90,
                   zorder=3, label="Original", marker="o")
    ax_lol.scatter(new,  y, color=cc["fill"], edgecolors=cc["edge"], s=110,
                   zorder=4, label="Repurposed", marker="D")

    for yi, d in zip(y, delt):
        ax_lol.text(max(new) + 1, yi, f"ΔS={d:+.1f}",
                    va="center", ha="left", fontsize=8, color="#000000")

    ylabels = [
        f"{_shorten(h['replaced_drug'],15)} → {_shorten(h['repurpose_drug_name'],15)}"
        for h in subset
    ]
    ax_lol.set_yticks(y)
    ax_lol.set_yticklabels(ylabels, fontsize=8, color="#000000")
    ax_lol.axvline(0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_lol.invert_yaxis()
    ax_lol.legend(fontsize=9, loc="lower right")
    ax_lol.set_xlabel("Loewe Synergy Score", fontsize=10, fontweight="bold",
                       color="#000000")
    ax_lol.set_title(f"Before → After Repurposing  [{cell_line}]",
                     fontsize=11, fontweight="bold", color="#000000")
    ax_lol.spines["top"].set_visible(False)
    ax_lol.spines["right"].set_visible(False)

    # ── Scatter ───────────────────────────────────────────────────────────────
    all_orig = [float(h["original_model_synergy"])       for h in hits
                if h["original_model_synergy"] != ""]
    all_new  = [float(h["estimated_repurposed_synergy"]) for h in hits
                if h["estimated_repurposed_synergy"] != ""]
    ax_sc.scatter(all_orig, all_new,
                  color=cc["fill"], edgecolors=cc["edge"], s=50, alpha=0.7, zorder=3)
    lim = [min(all_orig + all_new) - 5, max(all_orig + all_new) + 5]
    ax_sc.plot(lim, lim, "--", color="#888888", linewidth=1.2)
    ax_sc.set_xlim(lim); ax_sc.set_ylim(lim)
    ax_sc.set_xlabel("Original Synergy", fontsize=10, fontweight="bold",
                      color="#000000")
    ax_sc.set_ylabel("Repurposed Synergy", fontsize=10, fontweight="bold",
                      color="#000000")
    ax_sc.set_title(f"Orig vs Repurposed\n(n={len(all_orig)})",
                    fontsize=11, fontweight="bold", color="#000000")
    ax_sc.axhline(0, color="#CCCCCC", linewidth=0.8)
    ax_sc.axvline(0, color="#CCCCCC", linewidth=0.8)
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    ax_sc.tick_params(colors="#000000")

    fig.suptitle(
        f"Drug Repurposing Results — {cell_line}\n"
        "Each hit: replace one drug with an existing compound containing the optimal fragment",
        fontsize=12, fontweight="bold", color="#000000", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info("Combined panel → %s", outpath)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="eComb2 drug repurposing: find existing drugs containing "
                    "the optimal replacement fragment, with before/after synergy scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rescored",  required=True,
                   help="Path to rescored_reconstructed_candidates.csv.")
    p.add_argument("--frag-map",  required=True,
                   help="Path to smiles_to_fragments.pkl.")
    p.add_argument("--drug-map",  required=True,
                   help="Path to everything_drug_smiles_map.csv.")
    p.add_argument("--outdir",    default="repurposing",
                   help="Output directory.")
    p.add_argument("--cell-line", default=None,
                   help="Cell line label (used in filenames and plot titles).")
    p.add_argument("--tanimoto-threshold", type=float, default=0.5,
                   help="Minimum Tanimoto for tier-3 matches.")
    p.add_argument("--top-n",     type=int, default=0,
                   help="Keep only top-N hits globally (0 = all).")
    p.add_argument("--plot-top-n", type=int, default=20,
                   help="Number of hits to show in lollipop / bar plots.")
    p.add_argument("--min-delta", type=float, default=0.0,
                   help="Minimum rescored_delta_synergy_vs_model.")
    p.add_argument("--min-fragment-coverage", type=float, default=0.20,
                   help="SUBSTR tier: fragment must cover ≥ this fraction of drug atoms.")
    p.add_argument("--min-fragment-atoms", type=int, default=8,
                   help="SUBSTR tier: skip if candidate fragment has fewer heavy atoms.")
    p.add_argument("--exact-only", action="store_true", default=False,
                   help="Only EXACT fragment matches (highest quality, recommended).")
    p.add_argument("--no-plots",  action="store_true", default=False,
                   help="Skip plot generation.")
    # MoA filtering
    p.add_argument("--moa-cache", default=None,
                   help="Path to drug_moa_cache.csv (built by build_moa_cache.py). "
                        "When provided, annotates every hit with MoA data.")
    p.add_argument("--moa-filter", default=None,
                   choices=["high", "medium", "any", "none"],
                   help=(
                       "Filter hits by MoA match confidence. "
                       "'high'=same ChEMBL target, 'medium'=same ATC L1 code, "
                       "'any'=either, 'none'=no filter (annotate only). "
                       "Requires --moa-cache. Default: none (annotate, don't filter)."
                   ))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    frag_map          = load_fragment_map(args.frag_map)
    inv_index         = build_inverted_index(frag_map)
    smiles_to_name, _ = load_drug_name_map(args.drug_map)
    rescored          = load_rescored(args.rescored, args.min_delta)

    log.info("Running repurposing matching …")
    hits = match_candidates(
        rescored_rows=rescored,
        frag_map=frag_map,
        inv_index=inv_index,
        smiles_to_name=smiles_to_name,
        tanimoto_threshold=args.tanimoto_threshold,
        top_n=args.top_n,
        min_fragment_coverage=args.min_fragment_coverage,
        min_fragment_atoms=args.min_fragment_atoms,
        exact_only=args.exact_only,
    )
    log.info("Found %d repurposing hits", len(hits))

    # ── MoA annotation and optional filtering ─────────────────────────────────
    if args.moa_cache:
        moa_cache = load_moa_cache(args.moa_cache)
        hits = annotate_moa(hits, moa_cache)
        log.info("MoA annotation complete")

        if args.moa_filter and args.moa_filter != "none":
            before = len(hits)
            if args.moa_filter == "high":
                hits = [h for h in hits if h.get("moa_confidence") == "high"]
            elif args.moa_filter == "medium":
                hits = [h for h in hits
                        if h.get("moa_confidence") in ("high", "medium")]
            elif args.moa_filter == "any":
                hits = [h for h in hits
                        if h.get("moa_match") is True
                        or h.get("moa_confidence") == "unknown"]
            log.info("MoA filter '%s': %d → %d hits", args.moa_filter, before, len(hits))
    else:
        # Populate blank MoA columns so CSV is consistent
        for h in hits:
            for col in ("replaced_moa", "replaced_atc", "repurpose_moa",
                        "repurpose_atc", "moa_match_atc", "moa_match_target",
                        "moa_match", "moa_confidence"):
                h.setdefault(col, "")

    tag = f"_{args.cell_line}" if args.cell_line else ""
    os.makedirs(args.outdir, exist_ok=True)

    write_csv(
        os.path.join(args.outdir, f"repurposing_candidates{tag}.csv"),
        hits,
    )
    write_summary(
        os.path.join(args.outdir, f"repurposing_summary{tag}.txt"),
        hits, args,
    )

    if not args.no_plots and MPL_OK:
        cl = args.cell_line or "all"
        plot_lollipop(
            hits,
            os.path.join(args.outdir, f"repurposing_lollipop{tag}.svg"),
            cell_line=cl,
            top_n=args.plot_top_n,
        )
        plot_scatter(
            hits,
            os.path.join(args.outdir, f"repurposing_scatter{tag}.svg"),
            cell_line=cl,
        )
        plot_delta_bars(
            hits,
            os.path.join(args.outdir, f"repurposing_topbars{tag}.svg"),
            cell_line=cl,
            top_n=args.plot_top_n,
        )
        plot_synergy_panels(
            hits,
            os.path.join(args.outdir, f"repurposing_panels{tag}.svg"),
            cell_line=cl,
            top_n=args.plot_top_n,
        )
        log.info("Plots written to %s/", args.outdir)
    elif not MPL_OK:
        log.warning("matplotlib not available — skipping plots.")


if __name__ == "__main__":
    main()
