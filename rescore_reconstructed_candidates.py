#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Draw

from drug_graph_autoencoder import (
    GraphAutoencoder,
    collate_graphs,
    load_autoencoder_checkpoint,
    smiles_to_graph,
)
from synergy_prediction import DEFAULT_DRUG_SMILES_MAP, DEFAULT_FRAGMENT_EMBEDDINGS, DEFAULT_FRAGMENT_MAP, DEFAULT_MOLECULE_EMBEDDINGS
from synergy_shap_analysis import (
    ExampleRecord,
    build_examples_bundle,
    build_model,
    filter_examples,
    load_checkpoint,
    predict_single_with_gates,
    select_example,
)


MERGE_KEYS = [
    "drug_a",
    "drug_b",
    "replace_drug_smiles",
    "replace_drug_marked_smiles",
    "replace_drug_id",
    "source_fragment_smiles",
    "candidate_fragment_smiles",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore externally reconstructed replacement full SMILES using the saved "
            "synergy model, a saved molecule encoder checkpoint, and fragment embeddings."
        )
    )
    parser.add_argument("scored_csv", help="CSV containing reconstructed full_smiles, typically *_compact_scored.csv.")
    parser.add_argument(
        "--context-csv",
        required=True,
        help=(
            "Context CSV containing the original symbolic candidate metadata. "
            "Use replacement_candidates_ranked_with_fragment_sdf.csv or similar."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Saved synergy model checkpoint.")
    parser.add_argument("--dataset", default=None, help="Dataset for rebuilding pair examples.")
    parser.add_argument("--molecule-embeddings", default=DEFAULT_MOLECULE_EMBEDDINGS)
    parser.add_argument("--fragment-embeddings", default=DEFAULT_FRAGMENT_EMBEDDINGS)
    parser.add_argument("--fragment-map", default=DEFAULT_FRAGMENT_MAP)
    parser.add_argument("--drug-smiles-map", default=DEFAULT_DRUG_SMILES_MAP)
    parser.add_argument("--molecule-encoder-checkpoint", required=True)
    parser.add_argument("--fragment-encoder-checkpoint", default=None)
    parser.add_argument("--cell-line", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--variant",
        default="candidate_fragment",
        help="Which reconstructed rows to score from scored_csv. Defaults to candidate_fragment.",
    )
    return parser.parse_args()


def read_and_merge(scored_csv: Path, context_csv: Path, cell_line: str, variant: str) -> pd.DataFrame:
    scored_df = pd.read_csv(scored_csv)
    context_df = pd.read_csv(context_csv)

    missing_scored = [column for column in MERGE_KEYS + ["full_smiles"] if column not in scored_df.columns]
    if missing_scored:
        raise ValueError(f"Scored CSV is missing required columns: {missing_scored}")

    missing_context = [column for column in MERGE_KEYS if column not in context_df.columns]
    if missing_context:
        raise ValueError(f"Context CSV is missing merge columns: {missing_context}")

    if "cell_line_name" in context_df.columns:
        context_df = context_df[context_df["cell_line_name"].astype(str).str.lower() == str(cell_line).lower()].copy()

    if "variant" in scored_df.columns:
        scored_df = scored_df[scored_df["variant"].astype(str) == str(variant)].copy()

    if "error" in scored_df.columns:
        scored_df = scored_df[scored_df["error"].fillna("").astype(str).str.strip() == ""].copy()

    scored_df = scored_df[scored_df["full_smiles"].fillna("").astype(str).str.strip() != ""].copy()
    scored_df = scored_df[~scored_df["full_smiles"].astype(str).str.contains(r"\.", regex=True)].copy()
    if scored_df.empty:
        raise ValueError("No usable reconstructed rows remain after filtering by variant/error/full_smiles.")

    context_dedup = context_df.drop_duplicates(subset=MERGE_KEYS).copy()
    merged = scored_df.merge(context_dedup, on=MERGE_KEYS, how="inner", suffixes=("_scored", ""))
    if merged.empty:
        raise ValueError(
            "No reconstructed rows matched the context CSV. "
            "Make sure the scored CSV and context CSV were generated from the same symbolic run."
        )
    return merged.reset_index(drop=True)


def load_graph_encoder(path: Path, device: torch.device) -> GraphAutoencoder:
    model, payload = load_autoencoder_checkpoint(path, device)
    if str(payload.get("model_type", "")).strip().lower() != "graph":
        raise ValueError(f"Checkpoint {path} is not a graph autoencoder checkpoint.")
    return model  # type: ignore[return-value]


def encode_smiles(smiles: str, encoder: GraphAutoencoder, device: torch.device) -> np.ndarray:
    graph = smiles_to_graph(smiles, smiles)
    batch = collate_graphs([graph])
    features = batch["features"].to(device)
    adjacency = batch["adjacency"].to(device)
    mask = batch["mask"].to(device)
    with torch.no_grad():
        _, graph_embeddings = encoder.encode(features, adjacency, mask)
    return graph_embeddings[0].detach().cpu().numpy().astype(np.float32)


def get_or_infer_molecule_embedding(
    smiles: str,
    lookup: Dict[str, np.ndarray],
    encoder: GraphAutoencoder,
    device: torch.device,
) -> np.ndarray:
    cached = lookup.get(smiles)
    if cached is not None:
        return np.asarray(cached, dtype=np.float32)
    return encode_smiles(smiles, encoder, device)


def get_or_infer_fragment_embedding(
    smiles: str,
    lookup: Dict[str, np.ndarray],
    encoder: Optional[GraphAutoencoder],
    device: torch.device,
) -> np.ndarray:
    cached = lookup.get(smiles)
    if cached is not None:
        return np.asarray(cached, dtype=np.float32)
    if encoder is None:
        raise KeyError(
            f"Fragment embedding for {smiles!r} is missing from the lookup and no --fragment-encoder-checkpoint was provided."
        )
    return encode_smiles(smiles, encoder, device)


def pick_pair_example(examples: Dict[str, object], model, cell_line: str, device: str, drug_a: str, drug_b: str) -> ExampleRecord:
    filter_args = argparse.Namespace(
        cell_line=cell_line,
        drug_a=None,
        drug_b=None,
        pair_order_sensitive=False,
        select_by="predicted_top",
        example_index=0,
        device=device,
    )
    filtered = filter_examples(examples, filter_args, model)
    pair_rows = filtered[
        (filtered["drug_a"].astype(str).str.lower() == str(drug_a).lower())
        & (filtered["drug_b"].astype(str).str.lower() == str(drug_b).lower())
    ]
    if pair_rows.empty:
        swapped = filtered[
            (filtered["drug_a"].astype(str).str.lower() == str(drug_b).lower())
            & (filtered["drug_b"].astype(str).str.lower() == str(drug_a).lower())
        ]
        if swapped.empty:
            raise ValueError(f"Pair not found in cell line {cell_line}: {drug_a} + {drug_b}")
        pair_rows = swapped
    pair_rows = pair_rows.sort_values("predicted_synergy", ascending=False).reset_index(drop=True)
    return select_example(examples, pair_rows, argparse.Namespace(select_by="index", example_index=0))


def replace_fragment_row(fragment_matrix: np.ndarray, fragment_index: int, replacement: np.ndarray) -> np.ndarray:
    updated = np.asarray(fragment_matrix, dtype=np.float32).copy()
    if fragment_index < 0 or fragment_index >= updated.shape[0]:
        raise IndexError(f"fragment_index {fragment_index} is out of bounds for fragment matrix with {updated.shape[0]} rows")
    updated[fragment_index] = np.asarray(replacement, dtype=np.float32)
    return updated


def build_replaced_example(
    example: ExampleRecord,
    replace_drug: str,
    fragment_index: int,
    candidate_fragment_embedding: np.ndarray,
    replaced_drug_molecule_embedding: np.ndarray,
) -> ExampleRecord:
    if replace_drug == "a":
        return ExampleRecord(
            cell=example.cell,
            mol_a=np.asarray(replaced_drug_molecule_embedding, dtype=np.float32),
            mol_b=example.mol_b,
            frag_a=replace_fragment_row(example.frag_a, fragment_index, candidate_fragment_embedding),
            frag_b=example.frag_b,
            target=example.target,
            meta_row=example.meta_row,
        )
    if replace_drug == "b":
        return ExampleRecord(
            cell=example.cell,
            mol_a=example.mol_a,
            mol_b=np.asarray(replaced_drug_molecule_embedding, dtype=np.float32),
            frag_a=example.frag_a,
            frag_b=replace_fragment_row(example.frag_b, fragment_index, candidate_fragment_embedding),
            target=example.target,
            meta_row=example.meta_row,
        )
    raise ValueError(f"Unsupported replace_drug value: {replace_drug}")


def plot_pairwise_scores(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return

    ordered = df.copy().sort_values("rescored_delta_synergy_vs_observed", ascending=False).reset_index(drop=True)
    labels = [
        f"{row.drug_a}+{row.drug_b} | {row.candidate_fragment_smiles}"
        for row in ordered.itertuples(index=False)
    ]
    y = np.arange(len(ordered))

    plt.figure(figsize=(12, max(6, 0.45 * len(ordered))))
    plt.hlines(y, ordered["original_observed_synergy"], ordered["replaced_predicted_synergy"], color="#9e9e9e", lw=2)
    plt.scatter(ordered["original_observed_synergy"], y, color="#4c78a8", s=45, label="original observed")
    plt.scatter(ordered["replaced_predicted_synergy"], y, color="#f58518", s=45, label="replaced predicted")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Synergy score")
    plt.title("Original observed vs replaced predicted synergy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def constant_and_replaced_names(row: pd.Series) -> Tuple[str, str]:
    if str(row["replace_drug"]).lower() == "a":
        return str(row["drug_b"]), str(row["drug_a"])
    return str(row["drug_a"]), str(row["drug_b"])


def plot_drug_name_delta_bars(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return

    ordered = df.copy().sort_values("rescored_delta_synergy_vs_observed", ascending=False).reset_index(drop=True)
    labels = []
    for row in ordered.itertuples(index=False):
        constant_name, replaced_name = constant_and_replaced_names(pd.Series(row._asdict()))
        labels.append(f"{constant_name} + {replaced_name}")

    y = np.arange(len(ordered))
    colors = ["#1b9e77" if value >= 0 else "#d95f02" for value in ordered["rescored_delta_synergy_vs_observed"]]

    plt.figure(figsize=(11, max(6, 0.4 * len(ordered))))
    plt.barh(y, ordered["rescored_delta_synergy_vs_observed"], color=colors)
    plt.axvline(0.0, color="#666666", lw=1)
    plt.yticks(y, labels, fontsize=9)
    plt.xlabel("Replaced predicted - original observed synergy")
    plt.title("Synergy change after replacement")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_molecule_comparison_grid(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return

    molecules = []
    legends = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        replace_drug = str(row_dict["replace_drug"]).lower()
        if replace_drug == "a":
            constant_name = str(row_dict["drug_b"])
            constant_smiles = str(row_dict["drug_b_smiles"])
            original_name = str(row_dict["drug_a"])
            original_smiles = str(row_dict["drug_a_smiles"])
        else:
            constant_name = str(row_dict["drug_a"])
            constant_smiles = str(row_dict["drug_a_smiles"])
            original_name = str(row_dict["drug_b"])
            original_smiles = str(row_dict["drug_b_smiles"])

        replaced_smiles = str(row_dict["full_smiles"])
        constant_mol = Chem.MolFromSmiles(constant_smiles)
        original_mol = Chem.MolFromSmiles(original_smiles)
        replaced_mol = Chem.MolFromSmiles(replaced_smiles)
        if constant_mol is None or original_mol is None or replaced_mol is None:
            continue

        molecules.extend([constant_mol, original_mol, constant_mol, replaced_mol])
        legends.extend(
            [
                f"Constant drug\n{constant_name}",
                f"Original drug\n{original_name}\nObserved={float(row_dict['original_observed_synergy']):+.2f}",
                f"Constant drug\n{constant_name}",
                f"Replaced drug\nPredicted={float(row_dict['replaced_predicted_synergy']):+.2f}",
            ]
        )

    if not molecules:
        return

    from PIL import Image as PILImage, ImageDraw as PILDraw

    sub_w, sub_h = 320, 260
    cols = 2
    n_rows = (len(molecules) + cols - 1) // cols
    grid = PILImage.new("RGB", (sub_w * cols, sub_h * n_rows), (255, 255, 255))
    for idx, (mol, legend) in enumerate(zip(molecules, legends)):
        cell_img = Draw.MolToImage(mol, size=(sub_w, sub_h - 40))
        col_pos = idx % cols
        row_pos = idx // cols
        grid.paste(cell_img, (col_pos * sub_w, row_pos * sub_h))
        draw = PILDraw.Draw(grid)
        text_x = col_pos * sub_w + 4
        text_y = row_pos * sub_h + sub_h - 38
        draw.text((text_x, text_y), legend.replace("\n", "  "), fill=(40, 40, 40))
    grid.save(out_path)


def save_split_molecule_comparison_grids(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    top_increase = df.sort_values("rescored_delta_synergy_vs_observed", ascending=False).head(10).reset_index(drop=True)
    top_decrease = df.sort_values("rescored_delta_synergy_vs_observed", ascending=True).head(10).reset_index(drop=True)

    save_molecule_comparison_grid(top_increase, outdir / "rescored_molecule_comparison_top_increase.png")
    save_molecule_comparison_grid(top_decrease, outdir / "rescored_molecule_comparison_top_decrease.png")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = load_checkpoint(Path(args.checkpoint), args.device)
    model = build_model(checkpoint, args.device)
    examples = build_examples_bundle(args, checkpoint)

    molecule_embeddings = examples["molecule_embeddings"] if "molecule_embeddings" in examples else None
    if molecule_embeddings is None:
        from synergy_prediction import load_embedding_pickle

        molecule_embeddings = load_embedding_pickle(Path(args.molecule_embeddings))
    fragment_embeddings = examples["fragment_embeddings"]

    molecule_encoder = load_graph_encoder(Path(args.molecule_encoder_checkpoint), device)
    fragment_encoder = (
        load_graph_encoder(Path(args.fragment_encoder_checkpoint), device)
        if args.fragment_encoder_checkpoint
        else None
    )

    merged_df = read_and_merge(Path(args.scored_csv), Path(args.context_csv), args.cell_line, args.variant)

    results: List[Dict[str, object]] = []
    for row in merged_df.itertuples(index=False):
        example = pick_pair_example(
            examples=examples,
            model=model,
            cell_line=args.cell_line,
            device=args.device,
            drug_a=row.drug_a,
            drug_b=row.drug_b,
        )
        original_pred = predict_single_with_gates(model, example, np.ones(5, dtype=np.float32), args.device)
        original_observed = float(example.target)

        new_mol_embedding = get_or_infer_molecule_embedding(
            smiles=str(row.full_smiles),
            lookup=molecule_embeddings,
            encoder=molecule_encoder,
            device=device,
        )
        new_frag_embedding = get_or_infer_fragment_embedding(
            smiles=str(row.candidate_fragment_smiles),
            lookup=fragment_embeddings,
            encoder=fragment_encoder,
            device=device,
        )

        replaced_example = build_replaced_example(
            example=example,
            replace_drug=str(row.replace_drug).lower(),
            fragment_index=int(row.fragment_index),
            candidate_fragment_embedding=new_frag_embedding,
            replaced_drug_molecule_embedding=new_mol_embedding,
        )
        replaced_pred = predict_single_with_gates(model, replaced_example, np.ones(5, dtype=np.float32), args.device)

        results.append(
            {
                "cell_line_name": args.cell_line,
                "drug_a": row.drug_a,
                "drug_b": row.drug_b,
                "drug_a_smiles": row.drug_a_smiles,
                "drug_b_smiles": row.drug_b_smiles,
                "replace_drug": row.replace_drug,
                "selection": getattr(row, "selection", ""),
                "fragment_index": int(row.fragment_index),
                "replace_drug_smiles": row.replace_drug_smiles,
                "replace_drug_marked_smiles": row.replace_drug_marked_smiles,
                "replace_drug_id": row.replace_drug_id,
                "source_fragment_smiles": row.source_fragment_smiles,
                "candidate_fragment_smiles": row.candidate_fragment_smiles,
                "full_smiles": row.full_smiles,
                "variant": getattr(row, "variant", ""),
                "strain_energy": getattr(row, "strain_energy", np.nan),
                "force_field": getattr(row, "force_field", ""),
                "teacher_original_synergy": float(getattr(row, "teacher_original_synergy", np.nan)),
                "original_observed_synergy": original_observed,
                "original_model_predicted_synergy": float(original_pred),
                "replaced_predicted_synergy": float(replaced_pred),
                "rescored_delta_synergy_vs_observed": float(replaced_pred - original_observed),
                "rescored_delta_synergy_vs_model": float(replaced_pred - original_pred),
                "symbolic_pred_delta_synergy": float(getattr(row, "symbolic_pred_delta_synergy", np.nan)),
                "teacher_delta_synergy": float(getattr(row, "teacher_delta_synergy", np.nan)),
            }
        )

    results_df = pd.DataFrame(results).sort_values("rescored_delta_synergy_vs_observed", ascending=False).reset_index(drop=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(outdir / "rescored_reconstructed_candidates.csv", index=False)
    plot_pairwise_scores(results_df, outdir / "rescored_synergy_comparison.png")
    plot_drug_name_delta_bars(results_df, outdir / "rescored_synergy_delta_bars.png")
    save_molecule_comparison_grid(results_df, outdir / "rescored_molecule_comparison.png")
    save_split_molecule_comparison_grids(results_df, outdir)
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "cell_line_name": args.cell_line,
                "n_candidates_scored": int(len(results_df)),
                "variant": args.variant,
                "scored_csv": str(Path(args.scored_csv).resolve()),
                "context_csv": str(Path(args.context_csv).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "molecule_encoder_checkpoint": str(Path(args.molecule_encoder_checkpoint).resolve()),
                "fragment_encoder_checkpoint": None if args.fragment_encoder_checkpoint is None else str(Path(args.fragment_encoder_checkpoint).resolve()),
            },
            indent=2,
        )
    )

    print(f"Scored candidates: {len(results_df)}")
    print(f"Wrote: {outdir / 'rescored_reconstructed_candidates.csv'}")
    print(f"Wrote: {outdir / 'rescored_synergy_comparison.png'}")
    print(f"Wrote: {outdir / 'rescored_synergy_delta_bars.png'}")
    print(f"Wrote: {outdir / 'rescored_molecule_comparison.png'}")
    print(f"Wrote: {outdir / 'rescored_molecule_comparison_top_increase.png'}")
    print(f"Wrote: {outdir / 'rescored_molecule_comparison_top_decrease.png'}")


if __name__ == "__main__":
    main()
