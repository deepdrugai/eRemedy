#!/usr/bin/env python3
import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


SMILES_CANDIDATES = (
    "smiles",
    "canonical_smiles",
    "isosmiles",
    "structure",
)

DRUG_NAME_CANDIDATES = (
    "drug",
    "drug_name",
    "drugname",
    "compound",
    "compound_name",
    "agent",
    "name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read dataset pickle files, extract unique SMILES strings, and write each "
            "SMILES to numbered .smi files in per-dataset output folders."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=[
            "/media/exxact1/magesh/ecomb/dataset_EVERYTHING.pkl",
            "/media/exxact1/magesh/ecomb/dataset_ALMANAC.pkl",
            "/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl",
        ],
        help="Input dataset pickle paths.",
    )
    parser.add_argument(
        "--mapping-csv",
        default="/home/magesh/ecomb2/everything_drug_smiles_map.csv",
        help="Optional CSV with columns `drug` and `smiles` for drug-name-to-SMILES fallback.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional directory where per-dataset SMILES folders should be created.",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    if pd is None:
        raise RuntimeError(
            "pandas is required to read these pickle datasets. Run this script in an environment "
            "where pandas is installed."
        )


def normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {path}")

    with path.open("rb") as handle:
        return pickle.load(handle)


def flatten_frames(obj) -> List["pd.DataFrame"]:
    frames: List["pd.DataFrame"] = []

    if pd is not None and isinstance(obj, pd.DataFrame):
        frames.append(obj)
        return frames

    if isinstance(obj, dict):
        for value in obj.values():
            frames.extend(flatten_frames(value))
        return frames

    if isinstance(obj, (list, tuple)):
        for value in obj:
            frames.extend(flatten_frames(value))
        return frames

    return frames


def choose_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    for column in columns:
        column_lower = column.lower()
        if any(candidate in column_lower for candidate in candidates):
            return column

    return None


def extract_unique_smiles(obj) -> List[str]:
    frames = flatten_frames(obj)
    if not frames:
        raise ValueError(
            "No pandas DataFrame was found inside the pickle. Update `extract_unique_smiles` "
            "for this dataset schema."
        )

    unique_smiles = {}
    for frame in frames:
        columns = [str(column) for column in frame.columns]
        smiles_column = choose_column(columns, SMILES_CANDIDATES)
        if smiles_column is None:
            continue

        for value in frame[smiles_column].tolist():
            smiles = normalize_text(value)
            if smiles is None:
                continue
            unique_smiles.setdefault(smiles, None)

    if not unique_smiles:
        raise ValueError(
            "Could not find any SMILES column in the dataset. Expected one of "
            f"{SMILES_CANDIDATES} or a column containing those names."
        )

    return list(unique_smiles.keys())


def load_drug_smiles_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    mapping_df = pd.read_csv(path)
    required = {"drug", "smiles"}
    if not required.issubset(mapping_df.columns):
        raise ValueError(f"Mapping CSV must contain columns {required}: {path}")

    mapping = {}
    for drug, smiles_value in zip(mapping_df["drug"], mapping_df["smiles"]):
        drug_name = normalize_text(drug)
        smiles = normalize_text(smiles_value)
        if drug_name is None or smiles is None:
            continue
        mapping.setdefault(drug_name, smiles)
    return mapping


def extract_unique_smiles_from_drug_map(obj, drug_smiles_map: Dict[str, str]) -> List[str]:
    if not drug_smiles_map:
        raise ValueError("No SMILES column was found and no drug-to-SMILES mapping is available.")

    frames = flatten_frames(obj)
    if not frames:
        raise ValueError(
            "No pandas DataFrame was found inside the pickle. Update fallback extraction for this dataset schema."
        )

    unique_smiles = {}
    rows_kept = 0
    rows_dropped = 0

    for frame in frames:
        columns = [str(column) for column in frame.columns]
        drug_a_column = next((column for column in columns if column.lower() == "drug_a"), None)
        drug_b_column = next((column for column in columns if column.lower() == "drug_b"), None)
        if drug_a_column is None or drug_b_column is None:
            continue

        for drug_a, drug_b in zip(frame[drug_a_column].tolist(), frame[drug_b_column].tolist()):
            drug_a_name = normalize_text(drug_a)
            drug_b_name = normalize_text(drug_b)
            smiles_a = drug_smiles_map.get(drug_a_name) if drug_a_name else None
            smiles_b = drug_smiles_map.get(drug_b_name) if drug_b_name else None

            if smiles_a is None or smiles_b is None:
                rows_dropped += 1
                continue

            unique_smiles.setdefault(smiles_a, None)
            unique_smiles.setdefault(smiles_b, None)
            rows_kept += 1

    if not unique_smiles:
        raise ValueError(
            "Could not resolve any dataset rows through the drug-to-SMILES mapping. "
            "Expected `drug_a` and `drug_b` columns with names present in the mapping CSV."
        )

    print(f"  fallback mapping rows kept={rows_kept} dropped={rows_dropped}")
    return list(unique_smiles.keys())


def output_dir_for_dataset(path: Path, output_root: Optional[Path] = None) -> Path:
    if output_root is not None:
        return output_root / f"{path.stem}_SMILES"
    return path.with_name(f"{path.stem}_SMILES")


def write_smi_files(smiles_list: Iterable[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, smiles in enumerate(smiles_list, start=1):
        output_path = output_dir / f"{index}.smi"
        output_path.write_text(f"{smiles}\n", encoding="utf-8")
        count += 1
    return count


def process_dataset(
    dataset_path: Path,
    drug_smiles_map: Dict[str, str],
    output_root: Optional[Path],
) -> int:
    dataset = load_pickle(dataset_path)
    try:
        smiles_list = extract_unique_smiles(dataset)
    except ValueError as exc:
        message = str(exc)
        if "Could not find any SMILES column" not in message:
            raise
        smiles_list = extract_unique_smiles_from_drug_map(dataset, drug_smiles_map)
    return write_smi_files(smiles_list, output_dir_for_dataset(dataset_path, output_root))


def main() -> None:
    ensure_dependencies()
    args = parse_args()
    failures = []
    drug_smiles_map = load_drug_smiles_map(Path(args.mapping_csv).expanduser())
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else None

    for dataset_arg in args.datasets:
        dataset_path = Path(dataset_arg).expanduser().resolve()
        try:
            written = process_dataset(dataset_path, drug_smiles_map, output_root)
            print(
                f"{dataset_path.name}: wrote {written} SMILES to "
                f"{output_dir_for_dataset(dataset_path, output_root)}"
            )
        except Exception as exc:
            failures.append((dataset_path, exc))
            print(f"{dataset_path.name}: skipped ({exc})")

    if failures:
        print("\nDatasets skipped because no exportable SMILES were found:")
        for dataset_path, exc in failures:
            print(f"  - {dataset_path}: {exc}")


if __name__ == "__main__":
    main()
