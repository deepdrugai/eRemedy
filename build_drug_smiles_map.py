#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_V15_PATH = "/home/magesh/ecomb/samples_drugcomb_v15_with_compat.parquet"
DEFAULT_DRUG_INFO_PATH = "/home/magesh/ecomb/SynergyX_code/data/raw_data/drug_info.csv"
DEFAULT_DRUGCOMB_SMILES_PATH = "/home/magesh/ecomb/drugcombs_scored_smiles_blanks_removed.csv"


# Conservative synonym overrides justified by local repository evidence.
MANUAL_ALIASES = {
    "carboplatinum": "carboplatin",
    "eloxatin (tn)": "oxaliplatin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a drug-to-SMILES table for all unique drug_a/drug_b names in the "
            "DrugComb v1.5 parquet using local mapping sources."
        )
    )
    parser.add_argument("--v15", default=DEFAULT_V15_PATH, help="Path to samples_drugcomb_v15 parquet.")
    parser.add_argument(
        "--drug-info",
        default=DEFAULT_DRUG_INFO_PATH,
        help="Path to drug_info.csv containing alias columns and SMILES.",
    )
    parser.add_argument(
        "--drugcomb-smiles",
        default=DEFAULT_DRUGCOMB_SMILES_PATH,
        help="Path to DrugComb CSV with Drug1/Drug2 and DRUG1_SMILES/DRUG2_SMILES.",
    )
    parser.add_argument(
        "--output-csv",
        default="everything_drug_smiles_map.csv",
        help="Path to write the resolved drug-to-SMILES table.",
    )
    parser.add_argument(
        "--unresolved-csv",
        default="everything_drug_smiles_unresolved.csv",
        help="Path to write unresolved drug names.",
    )
    parser.add_argument(
        "--ambiguous-csv",
        default="everything_drug_smiles_ambiguous.csv",
        help="Path to write drugs with multiple candidate SMILES.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def clean_smiles(value: object) -> Optional[str]:
    text = normalize_text(value)
    if text is None:
        return None
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    return parts[0] if parts else None


def canonical_name_key(name: str) -> str:
    text = name.strip().lower()
    for token in (
        "hydrochloride",
        "hcl",
        "sulfate",
        "maleate",
        "mesylate",
        "tosylate",
        "phosphate",
        "sodium",
        "potassium",
        "hydrate",
        "free base",
        "(tn)",
    ):
        text = text.replace(token, " ")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return MANUAL_ALIASES.get(text, text)


def load_unique_drugs(v15_path: Path) -> List[str]:
    df = pd.read_parquet(v15_path)
    drugs = set(df["drug_a"].astype(str).str.strip()) | set(df["drug_b"].astype(str).str.strip())
    drugs.discard("")
    drugs.discard("nan")
    return sorted(drugs)


def build_drug_info_exact_map(drug_info_path: Path) -> Dict[str, Dict[str, str]]:
    drug_info = pd.read_csv(drug_info_path)
    smiles_col = "canonical_smi" if "canonical_smi" in drug_info.columns else "smiles"
    if smiles_col not in drug_info.columns:
        raise ValueError("drug_info.csv must contain canonical_smi or smiles.")

    name_cols = [column for column in ("dname", "drug", "drug_name") if column in drug_info.columns]
    if not name_cols:
        raise ValueError("drug_info.csv must contain at least one of dname/drug/drug_name.")

    mapping: Dict[str, Dict[str, str]] = {}
    for _, row in drug_info.iterrows():
        smiles = clean_smiles(row[smiles_col])
        if smiles is None:
            continue
        for name_col in name_cols:
            name = normalize_text(row[name_col])
            if name is None:
                continue
            mapping.setdefault(name, {"smiles": smiles, "source": f"drug_info:{name_col}:exact"})
    return mapping


def build_drugcomb_exact_map(drugcomb_smiles_path: Path) -> Dict[str, Dict[str, str]]:
    df = pd.read_csv(drugcomb_smiles_path)
    required = {"Drug1", "Drug2", "DRUG1_SMILES", "DRUG2_SMILES"}
    if not required.issubset(df.columns):
        raise ValueError("DrugComb scored SMILES CSV does not have the expected columns.")

    mapping: Dict[str, Dict[str, str]] = {}
    pairs = [
        ("Drug1", "DRUG1_SMILES"),
        ("Drug2", "DRUG2_SMILES"),
    ]
    for drug_col, smiles_col in pairs:
        for drug, smiles_value in zip(df[drug_col], df[smiles_col]):
            name = normalize_text(drug)
            smiles = clean_smiles(smiles_value)
            if name is None or smiles is None:
                continue
            mapping.setdefault(name, {"smiles": smiles, "source": f"drugcombs:{drug_col}:exact"})
    return mapping


def build_normalized_index(maps: Sequence[Dict[str, Dict[str, str]]]) -> Dict[str, List[Tuple[str, str, str]]]:
    index: Dict[str, List[Tuple[str, str, str]]] = {}
    for mapping in maps:
        for name, payload in mapping.items():
            key = canonical_name_key(name)
            if not key:
                continue
            entry = (name, payload["smiles"], payload["source"])
            index.setdefault(key, [])
            if entry not in index[key]:
                index[key].append(entry)
    return index


def resolve_drug(
    drug_name: str,
    exact_maps: Sequence[Dict[str, Dict[str, str]]],
    normalized_index: Dict[str, List[Tuple[str, str, str]]],
) -> Dict[str, object]:
    for mapping in exact_maps:
        if drug_name in mapping:
            payload = mapping[drug_name]
            return {
                "drug": drug_name,
                "smiles": payload["smiles"],
                "matched_name": drug_name,
                "source": payload["source"],
                "status": "resolved",
                "match_type": "exact",
                "candidate_count": 1,
            }

    key = canonical_name_key(drug_name)
    candidates = normalized_index.get(key, [])
    unique_smiles = sorted({candidate[1] for candidate in candidates})

    if len(unique_smiles) == 1 and candidates:
        matched_name, smiles, source = candidates[0]
        return {
            "drug": drug_name,
            "smiles": smiles,
            "matched_name": matched_name,
            "source": source,
            "status": "resolved",
            "match_type": "normalized",
            "candidate_count": len(candidates),
        }

    if len(unique_smiles) > 1:
        return {
            "drug": drug_name,
            "smiles": None,
            "matched_name": None,
            "source": None,
            "status": "ambiguous",
            "match_type": "normalized",
            "candidate_count": len(unique_smiles),
            "candidates": candidates,
        }

    return {
        "drug": drug_name,
        "smiles": None,
        "matched_name": None,
        "source": None,
        "status": "unresolved",
        "match_type": None,
        "candidate_count": 0,
    }


def build_outputs(
    drugs: Iterable[str],
    drug_info_map: Dict[str, Dict[str, str]],
    drugcomb_map: Dict[str, Dict[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_index = build_normalized_index([drug_info_map, drugcomb_map])

    resolved_rows = []
    unresolved_rows = []
    ambiguous_rows = []

    for drug in drugs:
        result = resolve_drug(drug, [drug_info_map, drugcomb_map], normalized_index)
        if result["status"] == "resolved":
            resolved_rows.append(result)
            continue
        if result["status"] == "ambiguous":
            candidates = result.pop("candidates")
            ambiguous_rows.append(result)
            for matched_name, smiles, source in candidates:
                ambiguous_rows.append(
                    {
                        "drug": drug,
                        "smiles": smiles,
                        "matched_name": matched_name,
                        "source": source,
                        "status": "ambiguous_candidate",
                        "match_type": "normalized",
                        "candidate_count": result["candidate_count"],
                    }
                )
            continue
        unresolved_rows.append(result)

    resolved_df = pd.DataFrame(resolved_rows).sort_values("drug").reset_index(drop=True)
    unresolved_df = pd.DataFrame(unresolved_rows).sort_values("drug").reset_index(drop=True)
    ambiguous_df = pd.DataFrame(ambiguous_rows).sort_values(
        ["drug", "status", "matched_name"], na_position="last"
    ).reset_index(drop=True)
    return resolved_df, unresolved_df, ambiguous_df


def main() -> None:
    args = parse_args()

    drugs = load_unique_drugs(Path(args.v15))
    drug_info_map = build_drug_info_exact_map(Path(args.drug_info))
    drugcomb_map = build_drugcomb_exact_map(Path(args.drugcomb_smiles))
    resolved_df, unresolved_df, ambiguous_df = build_outputs(drugs, drug_info_map, drugcomb_map)

    resolved_df.to_csv(args.output_csv, index=False)
    unresolved_df.to_csv(args.unresolved_csv, index=False)
    ambiguous_df.to_csv(args.ambiguous_csv, index=False)

    print(f"unique drugs: {len(drugs)}")
    print(f"resolved: {len(resolved_df)}")
    print(f"unresolved: {len(unresolved_df)}")
    print(f"ambiguous groups: {ambiguous_df['drug'].nunique() if not ambiguous_df.empty else 0}")
    print(f"wrote resolved map: {Path(args.output_csv).resolve()}")
    print(f"wrote unresolved list: {Path(args.unresolved_csv).resolve()}")
    print(f"wrote ambiguous list: {Path(args.ambiguous_csv).resolve()}")


if __name__ == "__main__":
    main()
