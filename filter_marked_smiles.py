#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


KEEP_COLUMNS = [
    "selection",
    "drug_a",
    "drug_b",
    "teacher_original_synergy",
    "replace_drug",
    "fragment_index",
    "replace_drug_smiles",
    "replace_drug_marked_smiles",
    "replace_drug_id",
    "source_fragment_smiles",
    "candidate_fragment_smiles",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the compact marked-SMILES subset used for downstream "
            "full-molecule reconstruction."
        )
    )
    parser.add_argument("input_csv", help="Annotated top-candidates CSV to read.")
    parser.add_argument("output_csv", help="Compact CSV to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()

    with input_path.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_path}")

        with output_path.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=KEEP_COLUMNS)
            writer.writeheader()
            for row in reader:
                writer.writerow({column: row.get(column, "") for column in KEEP_COLUMNS})


if __name__ == "__main__":
    main()
