#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map each parent SMILES from a numbered .smi folder to fragment SMILES from "
            "matching BRICS/linker SDF files."
        )
    )
    parser.add_argument(
        "--smiles-dir",
        default="dataset_EVERYTHING_SMILES",
        help="Folder containing numbered .smi files such as 1.smi, 2.smi, ...",
    )
    parser.add_argument(
        "--fragments-dir",
        default="dataset-smi-frag",
        help="Folder containing fragment SDF files such as b-1-000.sdf and l-1-000.sdf.",
    )
    parser.add_argument(
        "--output",
        default="smiles_to_fragments.pkl",
        help="Output pickle path for the SMILES -> fragment SMILES mapping.",
    )
    return parser.parse_args()


def ensure_rdkit() -> None:
    if Chem is None:
        raise RuntimeError("RDKit is required to read fragment SDF files.")


def read_parent_smiles(smi_path: Path) -> Optional[str]:
    text = smi_path.read_text(encoding="utf-8").strip()
    return text or None


def fragment_sdf_paths(fragments_dir: Path, molecule_id: str) -> List[Path]:
    paths = list(fragments_dir.glob(f"b-{molecule_id}-*.sdf"))
    paths.extend(fragments_dir.glob(f"l-{molecule_id}-*.sdf"))
    return sorted(paths)


def sdf_to_smiles(sdf_path: Path) -> Optional[str]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    if not supplier:
        return None
    mol = supplier[0]
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def build_mapping(smiles_dir: Path, fragments_dir: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}

    for smi_path in sorted(smiles_dir.glob("*.smi"), key=lambda path: int(path.stem)):
        molecule_id = smi_path.stem
        parent_smiles = read_parent_smiles(smi_path)
        if parent_smiles is None:
            continue

        fragment_smiles: List[str] = []
        seen = set()

        for sdf_path in fragment_sdf_paths(fragments_dir, molecule_id):
            smiles = sdf_to_smiles(sdf_path)
            if smiles is None or smiles in seen:
                continue
            fragment_smiles.append(smiles)
            seen.add(smiles)

        mapping[parent_smiles] = fragment_smiles

    return mapping


def main() -> None:
    ensure_rdkit()
    args = parse_args()

    smiles_dir = Path(args.smiles_dir).expanduser().resolve()
    fragments_dir = Path(args.fragments_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not smiles_dir.exists():
        raise FileNotFoundError(f"SMILES folder not found: {smiles_dir}")
    if not fragments_dir.exists():
        raise FileNotFoundError(f"Fragments folder not found: {fragments_dir}")

    mapping = build_mapping(smiles_dir, fragments_dir)
    with output_path.open("wb") as handle:
        pickle.dump(mapping, handle)

    print(f"mapped molecules: {len(mapping)}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
