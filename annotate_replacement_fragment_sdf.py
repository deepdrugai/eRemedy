#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from collections import defaultdict

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "RDKit is required. Run this script in an environment with rdkit installed."
    ) from exc


def canonicalize_smiles(smiles: str) -> str:
    smiles = (smiles or "").strip()
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def load_drug_smiles(smiles_dir: str):
    raw_to_ids = defaultdict(list)
    canon_to_ids = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(smiles_dir, "*.smi"))):
        drug_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as handle:
            raw = handle.read().strip()
        if not raw:
            continue
        raw_to_ids[raw].append(drug_id)
        canon = canonicalize_smiles(raw)
        if canon:
            canon_to_ids[canon].append(drug_id)
    return raw_to_ids, canon_to_ids


def load_fragment_index(fragment_dirs):
    fragment_index = defaultdict(list)
    for fragment_dir in fragment_dirs:
        pattern = os.path.join(fragment_dir, "[bl]-*-*.sdf")
        for path in sorted(glob.glob(pattern)):
            name = os.path.basename(path)
            prefix, drug_id, _ = name[:-4].split("-", 2)
            mol = Chem.SDMolSupplier(path, removeHs=False)[0]
            if mol is None:
                continue
            fragment_smiles = Chem.MolToSmiles(mol)
            fragment_index[(drug_id, fragment_smiles)].append((prefix, path))
    return fragment_index


def resolve_drug_id(replace_drug_smiles, raw_to_ids, canon_to_ids):
    raw_matches = raw_to_ids.get(replace_drug_smiles, [])
    if len(raw_matches) == 1:
        return raw_matches[0], "exact_raw"
    if len(raw_matches) > 1:
        return raw_matches[0], f"ambiguous_raw:{'|'.join(raw_matches)}"

    canon = canonicalize_smiles(replace_drug_smiles)
    canon_matches = canon_to_ids.get(canon, []) if canon else []
    if len(canon_matches) == 1:
        return canon_matches[0], "canonical"
    if len(canon_matches) > 1:
        return canon_matches[0], f"ambiguous_canonical:{'|'.join(canon_matches)}"
    return "", "drug_smiles_not_found"


def resolve_fragment_path(drug_id, source_fragment_smiles, fragment_index):
    canon_fragment = canonicalize_smiles(source_fragment_smiles)
    if not canon_fragment:
        return "", "source_fragment_invalid"

    matches = fragment_index.get((drug_id, canon_fragment), [])
    unique_matches = []
    seen_names = set()
    for prefix, path in matches:
        basename = os.path.basename(path)
        if basename in seen_names:
            continue
        seen_names.add(basename)
        unique_matches.append((prefix, path))

    if len(unique_matches) == 1:
        return unique_matches[0][1], f"matched:{unique_matches[0][0]}"
    if len(unique_matches) > 1:
        prefixes = "|".join(prefix for prefix, _ in unique_matches)
        return unique_matches[0][1], f"duplicate_fragment_matches:{len(unique_matches)}:{prefixes}"
    return "", "fragment_not_found"


def resolve_candidate_fragment_paths(candidate_fragment_smiles, fragment_index):
    canon_fragment = canonicalize_smiles(candidate_fragment_smiles)
    if not canon_fragment:
        return "", "candidate_fragment_invalid"

    matches = []
    for (drug_id, fragment_smiles), fragment_matches in fragment_index.items():
        if fragment_smiles != canon_fragment:
            continue
        matches.extend(fragment_matches)

    unique_matches = []
    seen_names = set()
    for prefix, path in matches:
        basename = os.path.basename(path)
        if basename in seen_names:
            continue
        seen_names.add(basename)
        unique_matches.append((prefix, path))

    if len(unique_matches) == 1:
        return unique_matches[0][1], f"matched:{unique_matches[0][0]}"
    if len(unique_matches) > 1:
        prefixes = "|".join(prefix for prefix, _ in unique_matches)
        joined_paths = ";".join(path for _, path in unique_matches)
        return joined_paths, f"multiple_matches:{len(unique_matches)}:{prefixes}"
    return "", "candidate_fragment_not_found"


def pick_fragment_match(parent_mol, fragment_mol):
    matches = parent_mol.GetSubstructMatches(fragment_mol, uniquify=True)
    if not matches:
        return None

    scored_matches = []
    for match in matches:
        match_set = set(match)
        cuts = []
        for atom_idx in match:
            atom = parent_mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in match_set:
                    continue
                bond = parent_mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                cuts.append((atom_idx, neighbor_idx, bond.GetBondType()))
        scored_matches.append((len(cuts), match, cuts))

    scored_matches.sort(key=lambda item: item[0])
    return scored_matches[0]


def build_marked_replace_drug_smiles(replace_drug_smiles, source_fragment_smiles):
    parent_mol = Chem.MolFromSmiles(replace_drug_smiles)
    fragment_mol = Chem.MolFromSmiles(source_fragment_smiles)
    if parent_mol is None:
        return "", "replace_drug_invalid"
    if fragment_mol is None:
        return "", "source_fragment_invalid"

    picked_match = pick_fragment_match(parent_mol, fragment_mol)
    if picked_match is None:
        return "", "source_fragment_no_substructure_match"

    cut_count, atom_match, cuts = picked_match
    if cut_count == 0:
        return "[*:1]", "whole_molecule_fragment"

    rw_mol = Chem.RWMol(parent_mol)
    for label, (_, external_atom_idx, bond_type) in enumerate(cuts, start=1):
        dummy_atom = Chem.Atom(0)
        dummy_atom.SetAtomMapNum(label)
        dummy_idx = rw_mol.AddAtom(dummy_atom)
        rw_mol.AddBond(external_atom_idx, dummy_idx, bond_type)

    for atom_idx in sorted(atom_match, reverse=True):
        rw_mol.RemoveAtom(atom_idx)

    marked_mol = rw_mol.GetMol()
    Chem.SanitizeMol(marked_mol)
    status = f"matched_substructure;attachment_points:{cut_count}"
    return Chem.MolToSmiles(marked_mol), status


def annotate_rows(rows, raw_to_ids, canon_to_ids, fragment_index):
    annotated = []
    for row in rows:
        replace_drug = row["replace_drug"].strip().lower()
        replace_drug_smiles = (
            row["drug_a_smiles"] if replace_drug == "a" else row["drug_b_smiles"]
        ).strip()
        drug_id, drug_status = resolve_drug_id(replace_drug_smiles, raw_to_ids, canon_to_ids)
        fragment_sdf, fragment_status = ("", "drug_id_missing")
        if drug_id:
            fragment_sdf, fragment_status = resolve_fragment_path(
                drug_id, row["source_fragment_smiles"], fragment_index
            )
        candidate_fragment_sdf, candidate_fragment_status = resolve_candidate_fragment_paths(
            row["candidate_fragment_smiles"], fragment_index
        )
        marked_smiles, marked_smiles_status = build_marked_replace_drug_smiles(
            replace_drug_smiles, row["source_fragment_smiles"]
        )

        annotated_row = dict(row)
        annotated_row["replace_drug_smiles"] = replace_drug_smiles
        annotated_row["replace_drug_id"] = drug_id
        annotated_row["replace_drug_marked_smiles"] = marked_smiles
        annotated_row["replace_drug_marked_smiles_status"] = marked_smiles_status
        annotated_row["fragment_sdf"] = fragment_sdf
        annotated_row["fragment_match_status"] = f"{drug_status};{fragment_status}"
        annotated_row["source_fragment_sdf"] = fragment_sdf
        annotated_row["source_fragment_match_status"] = annotated_row["fragment_match_status"]
        annotated_row["candidate_fragment_sdf"] = candidate_fragment_sdf
        annotated_row["candidate_fragment_match_status"] = candidate_fragment_status
        annotated.append(annotated_row)
    return annotated


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_top_subset(rows, score_column, top_n, top_per_pair=0):
    """Select the top-N candidates by *score_column*.

    When *top_per_pair* > 0 stratified sampling is applied first: up to
    *top_per_pair* best candidates per unique (drug_a, drug_b) pair are
    kept in a pool, and the global top-N is then taken from that pool.
    This guarantees that already-synergistic pairs (positive
    teacher_original_synergy) are not entirely crowded out by deeply
    antagonistic pairs that happen to show larger absolute improvements.
    """
    scored = [row for row in rows if row.get(score_column, "").strip()]
    scored = [row for row in scored if float(row[score_column]) > 0.0]
    scored = [row for row in scored if row.get("replace_drug_marked_smiles", "").strip()]
    scored.sort(key=lambda row: float(row[score_column]), reverse=True)

    if top_per_pair > 0:
        per_pair_counts = defaultdict(int)
        pool = []
        for row in scored:
            pair_key = (row.get("drug_a", ""), row.get("drug_b", ""))
            if per_pair_counts[pair_key] < top_per_pair:
                pool.append(row)
                per_pair_counts[pair_key] += 1
        scored = pool

    selected = []
    for row in scored[:top_n]:
        orig_syn = row.get("teacher_original_synergy", "")
        try:
            selection_label = "amplify" if float(orig_syn) > 0 else "rescue"
        except (ValueError, TypeError):
            selection_label = "positive"
        selected.append(
            {
                "selection": selection_label,
                "drug_a": row.get("drug_a", ""),
                "drug_b": row.get("drug_b", ""),
                "teacher_original_synergy": orig_syn,
                "teacher_delta_synergy": row.get("teacher_delta_synergy", ""),
                "drug_a_smiles": row["drug_a_smiles"],
                "drug_b_smiles": row["drug_b_smiles"],
                "replace_drug": row["replace_drug"],
                "replace_drug_smiles": row["replace_drug_smiles"],
                "replace_drug_id": row["replace_drug_id"],
                "replace_drug_marked_smiles": row["replace_drug_marked_smiles"],
                "replace_drug_marked_smiles_status": row["replace_drug_marked_smiles_status"],
                "fragment_index": row["fragment_index"],
                "source_fragment_smiles": row["source_fragment_smiles"],
                "fragment_sdf": row["fragment_sdf"],
                "source_fragment_sdf": row["source_fragment_sdf"],
                "source_fragment_match_status": row["source_fragment_match_status"],
                "candidate_fragment_smiles": row["candidate_fragment_smiles"],
                "candidate_fragment_sdf": row["candidate_fragment_sdf"],
                "candidate_fragment_match_status": row["candidate_fragment_match_status"],
            }
        )
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Annotate replacement candidate rows with the matching replace-drug fragment SDF."
        )
    )
    parser.add_argument("input_csv", help="Path to replacement_candidates_ranked.csv")
    parser.add_argument("--smiles-dir", required=True, help="Directory of *.smi drug files")
    parser.add_argument(
        "--fragment-dir",
        dest="fragment_dirs",
        action="append",
        required=True,
        help="Fragment directory containing b-*-*.sdf and l-*-*.sdf files. Pass more than once if needed.",
    )
    parser.add_argument("--output-csv", required=True, help="Path for the annotated full CSV")
    parser.add_argument(
        "--top-output-csv",
        help="Optional path for a top-N increase/decrease subset CSV",
    )
    parser.add_argument(
        "--score-column",
        default="teacher_delta_synergy",
        help="Column used to rank increases and decreases",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top increases and decreases to export",
    )
    parser.add_argument(
        "--top-per-pair",
        type=int,
        default=0,
        help=(
            "If > 0, apply stratified sampling before the global top-N cut: "
            "keep at most this many candidates per unique (drug_a, drug_b) pair. "
            "Guarantees that already-synergistic pairs are not crowded out by "
            "deeply antagonistic ones. 0 (default) disables stratification."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
        input_fieldnames = list(rows[0].keys()) if rows else []

    raw_to_ids, canon_to_ids = load_drug_smiles(args.smiles_dir)
    fragment_index = load_fragment_index(args.fragment_dirs)
    annotated = annotate_rows(rows, raw_to_ids, canon_to_ids, fragment_index)

    output_fieldnames = input_fieldnames + [
        "replace_drug_smiles",
        "replace_drug_id",
        "replace_drug_marked_smiles",
        "replace_drug_marked_smiles_status",
        "fragment_sdf",
        "fragment_match_status",
        "source_fragment_sdf",
        "source_fragment_match_status",
        "candidate_fragment_sdf",
        "candidate_fragment_match_status",
    ]
    write_csv(args.output_csv, annotated, output_fieldnames)

    if args.top_output_csv:
        top_rows = build_top_subset(annotated, args.score_column, args.top_n, args.top_per_pair)
        top_fieldnames = [
            "selection",
            "drug_a",
            "drug_b",
            "teacher_original_synergy",
            "teacher_delta_synergy",
            "drug_a_smiles",
            "drug_b_smiles",
            "replace_drug",
            "replace_drug_smiles",
            "replace_drug_id",
            "replace_drug_marked_smiles",
            "replace_drug_marked_smiles_status",
            "fragment_index",
            "source_fragment_smiles",
            "fragment_sdf",
            "source_fragment_sdf",
            "source_fragment_match_status",
            "candidate_fragment_smiles",
            "candidate_fragment_sdf",
            "candidate_fragment_match_status",
        ]
        write_csv(args.top_output_csv, top_rows, top_fieldnames)


if __name__ == "__main__":
    main()
