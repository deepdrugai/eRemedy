"""
build_moa_cache.py
==================
Builds a mechanism-of-action (MoA) cache for every drug in the pipeline by
querying the ChEMBL REST API.  No extra libraries required — uses only Python's
standard library plus RDKit (for canonical SMILES and InChI-key generation).

Lookup strategy (triple fallback per drug):
  1. Preferred-name exact match  → fastest, works for most approved drugs
  2. InChI-key connectivity-layer prefix match  → bypasses stereo/salt variants
  3. SMILES flexmatch  → last resort for unusual representations

Output columns in drug_moa_cache.csv
--------------------------------------
  drug_name            input name from the drug SMILES map
  smiles               input SMILES
  chembl_id            resolved ChEMBL ID (empty = not found)
  lookup_method        name | inchikey_prefix | smiles_flex | not_found
  atc_classifications  comma-separated ATC codes (e.g. L01XK05)
  atc_l1               first ATC letter: L=antineoplastic, J=anti-infective …
  mechanism_of_action  semicolon-joined free-text MoA strings from ChEMBL
  target_chembl_ids    semicolon-joined ChEMBL target IDs
  action_types         semicolon-joined action types (INHIBITOR, AGONIST …)
  max_phase            highest clinical phase in ChEMBL

Usage
-----
  # Build cache for all drugs in the SMILES map:
  python build_moa_cache.py \\
      --drug-map  everything_drug_smiles_map.csv \\
      --outfile   drug_moa_cache.csv

  # Extend an existing cache (skips already-resolved entries):
  python build_moa_cache.py \\
      --drug-map  everything_drug_smiles_map.csv \\
      --outfile   drug_moa_cache.csv \\
      --resume

  # Only resolve the drugs actually present in repurposing results:
  python build_moa_cache.py \\
      --drug-map  everything_drug_smiles_map.csv \\
      --repurposing-csv  repurposing/A2058/repurposing_candidates_A2058.csv \\
      --outfile   drug_moa_cache.csv
"""

import argparse
import csv
import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple
import time
import urllib.parse
import urllib.request
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import MolToSmiles
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    log.warning("RDKit not available — InChI-key and SMILES-flex fallbacks disabled.")

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
HEADERS = {"Accept": "application/json", "User-Agent": "ecomb2-moa-cache/1.0"}
REQUEST_DELAY = 0.35   # seconds between API calls (stay well under rate limit)
MAX_RETRIES   = 3


# ─────────────────────────────────────────────────────────────────────────────
# ChEMBL helpers
# ─────────────────────────────────────────────────────────────────────────────

def chembl_get(url: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                log.debug("ChEMBL GET failed (%s): %s", url[:80], e)
                return {}
            time.sleep(1.5 * (attempt + 1))
    return {}


def inchikey_prefix(smi: str) -> Optional[str]:
    """Return the connectivity-layer prefix of the InChI key (first block)."""
    if not RDKIT_OK or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    inchi = MolToInchi(mol)
    if inchi is None:
        return None
    ik = InchiToInchiKey(inchi)
    return ik.split("-")[0] if ik else None


def canonical_smiles(smi: str) -> Optional[str]:
    if not RDKIT_OK or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    return MolToSmiles(mol) if mol else None


def _prefer_parent(mols: List[dict]) -> dict:
    """From a list of ChEMBL molecule entries, prefer the parent compound."""
    for m in mols:
        hier = m.get("molecule_hierarchy", {})
        if hier.get("parent_chembl_id") == m.get("molecule_chembl_id"):
            return m
    return mols[0]


def _mol_query(param: str, value: str) -> List[dict]:
    url = "%s/molecule.json?%s=%s&format=json" % (
        CHEMBL_BASE, param, urllib.parse.quote(value)
    )
    return chembl_get(url).get("molecules", [])


def _mechanism_query(chembl_id: str) -> List[dict]:
    url = "%s/mechanism.json?molecule_chembl_id=%s&format=json" % (
        CHEMBL_BASE, chembl_id
    )
    return chembl_get(url).get("mechanisms", [])


def lookup_chembl(drug_name: str, smi: str) -> dict:
    """
    Resolve a drug (by name + SMILES) to its ChEMBL record and MoA data.
    Returns a flat dict ready to write as a CSV row.
    """
    result = {
        "drug_name":           drug_name,
        "smiles":              smi,
        "chembl_id":           "",
        "lookup_method":       "not_found",
        "atc_classifications": "",
        "atc_l1":              "",
        "mechanism_of_action": "",
        "target_chembl_ids":   "",
        "action_types":        "",
        "max_phase":           "",
    }

    mols = []

    # 1. Preferred-name exact match
    mols = _mol_query("pref_name__iexact", drug_name)
    if mols:
        result["lookup_method"] = "name"
    
    # 2. Synonym search (catches trade names, abbreviations)
    if not mols:
        url = ("%s/molecule.json?molecule_synonyms__molecule_synonym__iexact=%s"
               "&format=json" % (CHEMBL_BASE, urllib.parse.quote(drug_name)))
        mols = chembl_get(url).get("molecules", [])
        if mols:
            result["lookup_method"] = "synonym"
        time.sleep(REQUEST_DELAY)

    # 3. InChI-key connectivity prefix (stereo / salt agnostic)
    if not mols:
        prefix = inchikey_prefix(smi)
        if prefix:
            mols = _mol_query(
                "molecule_structures__standard_inchi_key__startswith", prefix
            )
            if mols:
                result["lookup_method"] = "inchikey_prefix"
        time.sleep(REQUEST_DELAY)

    # 4. SMILES flexmatch (last resort)
    if not mols:
        canon = canonical_smiles(smi)
        if canon:
            mols = _mol_query(
                "molecule_structures__canonical_smiles__flexmatch", canon
            )
            if mols:
                result["lookup_method"] = "smiles_flex"
        time.sleep(REQUEST_DELAY)

    if not mols:
        return result

    mol_entry = _prefer_parent(mols)
    chembl_id = mol_entry["molecule_chembl_id"]
    result["chembl_id"]  = chembl_id
    result["max_phase"]  = str(mol_entry.get("max_phase", ""))

    atc_codes = mol_entry.get("atc_classifications", [])
    result["atc_classifications"] = ", ".join(atc_codes)
    result["atc_l1"] = atc_codes[0][0] if atc_codes else ""

    # Fetch mechanism of action
    time.sleep(REQUEST_DELAY)
    mechs = _mechanism_query(chembl_id)
    result["mechanism_of_action"] = "; ".join(m["mechanism_of_action"] for m in mechs)
    result["target_chembl_ids"]   = "; ".join(m.get("target_chembl_id", "") for m in mechs)
    result["action_types"]        = "; ".join(m.get("action_type", "") for m in mechs)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Drug collection
# ─────────────────────────────────────────────────────────────────────────────

def load_drug_smiles_map(csv_path: str) -> Dict[str, str]:
    drug_smi = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("drug", "").strip()
            smi  = row.get("smiles", "").strip()
            if name and smi:
                drug_smi[name] = smi
    return drug_smi


def collect_pipeline_drugs(
    drug_map: Dict[str, str],
    repurposing_csvs: List[str],
) -> Dict[str, str]:
    """
    If repurposing CSVs are provided, restrict to only the drugs that appear
    in the repurposing results (replaced_drug, partner_drug, repurpose_drug_name).
    Always resolves SMILES from the global drug_map.
    """
    if not repurposing_csvs:
        return drug_map

    needed: Set[str] = set()
    for path in repurposing_csvs:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                needed.update([
                    row.get("replaced_drug", ""),
                    row.get("partner_drug", ""),
                    row.get("repurpose_drug_name", ""),
                    row.get("original_drug_a", ""),
                    row.get("original_drug_b", ""),
                ])
    needed.discard("")

    # Map names → SMILES using the global drug map (canonical SMILES lookup)
    out: Dict[str, str] = {}
    for name in sorted(needed):
        if name in drug_map:
            out[name] = drug_map[name]
        else:
            # Fuzzy: case-insensitive match
            lower_map = {k.lower(): v for k, v in drug_map.items()}
            if name.lower() in lower_map:
                out[name] = lower_map[name.lower()]
            else:
                out[name] = ""   # SMILES unknown; will rely on name lookup only
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "drug_name", "smiles", "chembl_id", "lookup_method",
    "atc_classifications", "atc_l1",
    "mechanism_of_action", "target_chembl_ids", "action_types", "max_phase",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a ChEMBL MoA cache for pipeline drugs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--drug-map",  required=True,
                   help="Path to everything_drug_smiles_map.csv.")
    p.add_argument("--outfile",   default="drug_moa_cache.csv",
                   help="Output cache CSV path.")
    p.add_argument("--repurposing-csv", nargs="+", default=[],
                   help="Repurposing result CSVs; if given, only resolve "
                        "drugs that appear in these files.")
    p.add_argument("--resume",    action="store_true", default=False,
                   help="Skip drugs already present in the output file.")
    p.add_argument("--delay",     type=float, default=REQUEST_DELAY,
                   help="Seconds between ChEMBL API calls.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global REQUEST_DELAY
    REQUEST_DELAY = args.delay

    # ── Load drug SMILES map ──────────────────────────────────────────────────
    drug_map = load_drug_smiles_map(args.drug_map)
    log.info("Drug SMILES map: %d entries", len(drug_map))

    # ── Determine which drugs to resolve ─────────────────────────────────────
    to_resolve = collect_pipeline_drugs(drug_map, args.repurposing_csv)
    log.info("Drugs to resolve: %d", len(to_resolve))

    # ── Resume: skip already-cached entries ──────────────────────────────────
    already_done: Set[str] = set()
    if args.resume and os.path.exists(args.outfile):
        with open(args.outfile, newline="") as f:
            for row in csv.DictReader(f):
                already_done.add(row["drug_name"])
        log.info("Resume mode: %d already cached, skipping them", len(already_done))

    pending = {n: s for n, s in to_resolve.items() if n not in already_done}
    log.info("Pending lookups: %d", len(pending))

    # ── Write cache ───────────────────────────────────────────────────────────
    write_header = not (args.resume and os.path.exists(args.outfile))
    mode = "a" if (args.resume and os.path.exists(args.outfile)) else "w"

    with open(args.outfile, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, (name, smi) in enumerate(sorted(pending.items()), 1):
            log.info("[%d/%d] %s", i, len(pending), name)
            try:
                row = lookup_chembl(name, smi)
            except Exception as e:
                log.warning("Lookup failed for %s: %s", name, e)
                row = {f: "" for f in FIELDNAMES}
                row["drug_name"] = name
                row["smiles"]    = smi
                row["lookup_method"] = "error"
            writer.writerow(row)
            f.flush()
            time.sleep(REQUEST_DELAY)

    log.info("Cache written to %s", args.outfile)

    # ── Quick summary ─────────────────────────────────────────────────────────
    found = not_found = with_moa = 0
    with open(args.outfile, newline="") as f:
        for row in csv.DictReader(f):
            if row["chembl_id"]:
                found += 1
                if row["mechanism_of_action"]:
                    with_moa += 1
            else:
                not_found += 1
    log.info("Summary: %d resolved, %d not found, %d with MoA data",
             found, not_found, with_moa)


if __name__ == "__main__":
    main()
