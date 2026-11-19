#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from drug_graph_autoencoder import (
    Chem,
    GraphAutoencoder,
    MoleculeDataset,
    MoleculeGraph,
    collate_graphs,
    collect_embeddings,
    collect_unique_inference_smiles,
    load_autoencoder_checkpoint,
    normalize_text,
    read_smiles_text_file,
    save_autoencoder_checkpoint,
    save_embeddings,
    set_seed,
    smiles_to_graph,
    train_autoencoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a SMILES-to-fragments pickle, collect unique fragment SMILES, train a "
            "graph autoencoder on those fragments, and save learned fragment embeddings."
        )
    )
    parser.add_argument(
        "--input",
        default="smiles_to_fragments.pkl",
        help="Path to the pickle mapping parent SMILES -> list of fragment SMILES.",
    )
    parser.add_argument(
        "--output",
        default="fragment_smiles_embeddings.pkl",
        help="Path to the output pickle containing fragment SMILES embeddings.",
    )
    parser.add_argument(
        "--metadata-output",
        default="fragment_smiles_embeddings_metadata.json",
        help="Path to the output metadata JSON.",
    )
    parser.add_argument(
        "--checkpoint-output",
        default="fragment_smiles_encoder.pt",
        help="Path to save the trained fragment autoencoder checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-input",
        default=None,
        help="Optional fragment autoencoder checkpoint to load for inference without retraining.",
    )
    parser.add_argument(
        "--infer-smiles",
        nargs="*",
        default=None,
        help="Optional fragment SMILES strings to embed with --checkpoint-input.",
    )
    parser.add_argument(
        "--infer-smiles-file",
        default=None,
        help="Optional text file containing one fragment SMILES per line for inference.",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_fragment_mapping(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Fragment mapping file not found: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Expected the fragment mapping pickle to contain a dict.")
    return payload


def extract_unique_fragments(mapping: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    unique = {}
    for fragments in mapping.values():
        if not isinstance(fragments, list):
            continue
        for fragment_smiles in fragments:
            smiles = str(fragment_smiles).strip()
            if not smiles:
                continue
            unique.setdefault(smiles, smiles)
    if not unique:
        raise ValueError("No fragment SMILES were found in the mapping pickle.")
    return sorted(unique.items(), key=lambda item: item[0])


def build_graphs(unique_fragments: List[Tuple[str, str]]) -> Tuple[List[MoleculeGraph], List[Tuple[str, str, str]]]:
    graphs: List[MoleculeGraph] = []
    skipped: List[Tuple[str, str, str]] = []
    for fragment_id, smiles in unique_fragments:
        try:
            graphs.append(smiles_to_graph(fragment_id, smiles))
        except Exception as exc:
            skipped.append((fragment_id, smiles, str(exc)))
    return graphs, skipped


def run_inference(args: argparse.Namespace) -> None:
    if Chem is None:
        raise RuntimeError(
            "RDKit is required to convert fragment SMILES into molecular graphs. Install rdkit before running this script."
        )

    if not args.checkpoint_input:
        raise ValueError("Inference mode requires --checkpoint-input.")

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint_input).expanduser().resolve()
    model, checkpoint_payload = load_autoencoder_checkpoint(checkpoint_path, device)
    if str(checkpoint_payload.get("model_type", "")).strip().lower() != "graph":
        raise ValueError(
            "Fragment inference from SMILES is only supported for graph autoencoder checkpoints, "
            f"but {checkpoint_path} stores model_type={checkpoint_payload.get('model_type')!r}."
        )

    smiles_file = Path(args.infer_smiles_file).expanduser().resolve() if args.infer_smiles_file else None
    unique_fragments = collect_unique_inference_smiles(args.infer_smiles, smiles_file)
    print(f"Running fragment inference for {len(unique_fragments)} SMILES.")

    graphs, skipped = build_graphs(unique_fragments)
    if not graphs:
        raise ValueError("No valid fragment graphs could be created for inference.")

    embeddings, metadata = collect_embeddings(
        model=model,
        graphs=graphs,
        device=device,
        batch_size=args.batch_size,
    )

    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    save_embeddings(output_path, embeddings)
    metadata_payload = {
        "embedding_source": "fragment_smiles_graph_autoencoder_inference",
        "checkpoint_input": str(checkpoint_path),
        "checkpoint_output": None,
        "num_requested_fragments": len(unique_fragments),
        "num_embedded_fragments": len(graphs),
        "num_skipped_fragments": len(skipped),
        "embedding_dim": int(checkpoint_payload["embedding_dim"]),
        "entries": metadata,
        "skipped": [
            {"fragment_id": fragment_id, "smiles": smiles, "reason": reason}
            for fragment_id, smiles, reason in skipped
        ],
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))

    print(f"Saved fragment inference embeddings to {output_path}")
    print(f"Saved fragment inference metadata to {metadata_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.infer_smiles or args.infer_smiles_file:
        run_inference(args)
        return

    if Chem is None:
        raise RuntimeError(
            "RDKit is required to convert fragment SMILES into molecular graphs. Install rdkit before running this script."
        )

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    checkpoint_output_path = Path(args.checkpoint_output)
    device = torch.device(args.device)

    mapping = load_fragment_mapping(input_path)
    unique_fragments = extract_unique_fragments(mapping)
    print(f"Found {len(unique_fragments)} unique fragment SMILES.")

    graphs, skipped = build_graphs(unique_fragments)
    if not graphs:
        raise ValueError("No valid fragment graphs could be created from the mapping pickle.")

    if skipped:
        print(f"Skipped {len(skipped)} fragments due to invalid or unparsable SMILES.")

    train_loader = DataLoader(
        MoleculeDataset(graphs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )

    input_dim = graphs[0].node_features.shape[1]
    model = GraphAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
    ).to(device)

    train_autoencoder(
        model=model,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    embeddings, metadata = collect_embeddings(
        model=model,
        graphs=graphs,
        device=device,
        batch_size=args.batch_size,
    )

    metadata_payload = {
        "input_fragment_mapping": str(input_path),
        "embedding_source": "fragment_smiles_graph_autoencoder",
        "checkpoint_output": str(checkpoint_output_path.resolve()),
        "num_unique_fragments": len(unique_fragments),
        "num_trained_graphs": len(graphs),
        "num_skipped_graphs": len(skipped),
        "embedding_dim": args.embedding_dim,
        "entries": metadata,
        "skipped": [
            {"fragment_id": fragment_id, "smiles": smiles, "reason": reason}
            for fragment_id, smiles, reason in skipped
        ],
    }

    save_autoencoder_checkpoint(
        path=checkpoint_output_path,
        model=model,
        model_type="graph",
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        source_metadata={
            "input_fragment_mapping": str(input_path),
            "embedding_source": "fragment_smiles_graph_autoencoder",
        },
    )
    save_embeddings(output_path, embeddings)
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))

    print(f"Saved fragment embeddings to {output_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved checkpoint to {checkpoint_output_path}")


if __name__ == "__main__":
    main()
