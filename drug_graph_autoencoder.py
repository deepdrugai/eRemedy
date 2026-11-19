#!/usr/bin/env python3
import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


DRUG_NAME_CANDIDATES = (
    "drug",
    "drug_name",
    "drugname",
    "compound",
    "compound_name",
    "agent",
    "name",
)

SMILES_CANDIDATES = (
    "smiles",
    "canonical_smiles",
    "isosmiles",
    "structure",
)


@dataclass
class MoleculeGraph:
    drug_id: str
    smiles: str
    node_features: torch.Tensor
    adjacency: torch.Tensor


@dataclass
class FingerprintDrug:
    drug_id: str
    fingerprint: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a dataset pickle or a folder of .smi files, train a graph autoencoder "
            "from unique SMILES strings, and save learned embeddings."
        )
    )
    parser.add_argument(
        "--input",
        default="dataset_ONEIL.pkl",
        help="Path to the input pickle dataset. Ignored if --smiles-dir is set.",
    )
    parser.add_argument(
        "--smiles-dir",
        default=None,
        help="Optional folder containing .smi files. If set, unique SMILES are read from this folder.",
    )
    parser.add_argument(
        "--output",
        default="drug_smiles_embeddings.pkl",
        help="Path to the output embedding file. `.pkl` stores a dict keyed by SMILES or drug name.",
    )
    parser.add_argument(
        "--metadata-output",
        default="drug_smiles_embeddings_metadata.json",
        help="Optional JSON file with drug names and embedding indices.",
    )
    parser.add_argument(
        "--checkpoint-output",
        default="drug_smiles_encoder.pt",
        help="Path to save the trained autoencoder checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-input",
        default=None,
        help="Optional autoencoder checkpoint to load for inference without retraining.",
    )
    parser.add_argument(
        "--infer-smiles",
        nargs="*",
        default=None,
        help="Optional SMILES strings to embed with --checkpoint-input.",
    )
    parser.add_argument(
        "--infer-smiles-file",
        default=None,
        help="Optional text file containing one SMILES per line for inference.",
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dependencies() -> None:
    if pd is None:
        raise RuntimeError(
            "pandas is required to load the pickle dataset. Install it before running this script."
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


def read_smiles_dir(smiles_dir: Path) -> List[Tuple[str, str]]:
    if not smiles_dir.exists():
        raise FileNotFoundError(f"SMILES folder not found: {smiles_dir}")
    if not smiles_dir.is_dir():
        raise ValueError(f"SMILES path is not a directory: {smiles_dir}")

    unique_pairs: Dict[str, str] = {}
    for smi_path in sorted(smiles_dir.glob("*.smi"), key=lambda path: path.name):
        smiles = normalize_text(smi_path.read_text(encoding="utf-8"))
        if smiles is None:
            continue
        unique_pairs.setdefault(smiles, smiles)

    if not unique_pairs:
        raise ValueError(f"No valid SMILES were found in folder: {smiles_dir}")

    return sorted(unique_pairs.items(), key=lambda item: item[0])


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


def extract_unique_drugs(obj) -> List[Tuple[str, str]]:
    frames = flatten_frames(obj)
    if not frames:
        raise ValueError(
            "No pandas DataFrame was found inside the pickle. Update `extract_unique_drugs` for your dataset schema."
        )

    unique_pairs: Dict[str, str] = {}

    for frame in frames:
        columns = [str(column) for column in frame.columns]
        smiles_column = choose_column(columns, SMILES_CANDIDATES)
        if smiles_column is None:
            continue

        drug_column = choose_column(columns, DRUG_NAME_CANDIDATES)
        for row in frame.itertuples(index=False):
            row_dict = {columns[idx]: row[idx] for idx in range(len(columns))}
            smiles = normalize_text(row_dict.get(smiles_column))
            if smiles is None:
                continue

            drug_name = normalize_text(row_dict.get(drug_column)) if drug_column else None
            drug_id = drug_name or smiles
            unique_pairs.setdefault(drug_id, smiles)

    if not unique_pairs:
        raise ValueError(
            "Could not find any SMILES column in the dataset. Expected one of "
            f"{SMILES_CANDIDATES} or a column containing those names."
        )

    return sorted(unique_pairs.items(), key=lambda item: item[0])


def extract_unique_drugs_from_fingerprints(obj) -> List[FingerprintDrug]:
    if not isinstance(obj, dict):
        raise ValueError("Fingerprint fallback expects a dict-style dataset payload.")

    meta = obj.get("meta")
    fp_a = obj.get("fp_a")
    fp_b = obj.get("fp_b")
    if meta is None or fp_a is None or fp_b is None:
        raise ValueError("Fingerprint fallback requires `meta`, `fp_a`, and `fp_b` entries.")
    if pd is None or not isinstance(meta, pd.DataFrame):
        raise ValueError("Fingerprint fallback requires `meta` to be a pandas DataFrame.")
    if "drug_a" not in meta.columns or "drug_b" not in meta.columns:
        raise ValueError("Fingerprint fallback requires `drug_a` and `drug_b` columns in `meta`.")

    unique_drugs: Dict[str, np.ndarray] = {}
    for column_name, fp_key in (("drug_a", "fp_a"), ("drug_b", "fp_b")):
        fingerprints = np.asarray(obj[fp_key])
        for idx, drug_name in enumerate(meta[column_name].astype(str).tolist()):
            fingerprint = fingerprints[idx].astype(np.float32)
            previous = unique_drugs.get(drug_name)
            if previous is not None and not np.array_equal(previous, fingerprint):
                raise ValueError(
                    f"Drug `{drug_name}` maps to multiple fingerprints in `{fp_key}`."
                )
            unique_drugs[drug_name] = fingerprint

    if not unique_drugs:
        raise ValueError("No unique drugs could be reconstructed from `drug_a`/`drug_b` fingerprints.")

    return [
        FingerprintDrug(drug_id=drug_id, fingerprint=torch.from_numpy(fingerprint))
        for drug_id, fingerprint in sorted(unique_drugs.items(), key=lambda item: item[0])
    ]


def atom_features(atom: "Chem.Atom") -> List[float]:
    symbol_set = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    symbol = atom.GetSymbol()
    features = [1.0 if symbol == token else 0.0 for token in symbol_set]
    features.extend(
        [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
            float(atom.GetHybridization()),
            float(atom.IsInRing()),
        ]
    )
    return features


def smiles_to_graph(drug_id: str, smiles: str) -> MoleculeGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for drug `{drug_id}`: {smiles}")

    num_atoms = mol.GetNumAtoms()
    node_features = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    adjacency = np.zeros((num_atoms, num_atoms), dtype=np.float32)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    adjacency += np.eye(num_atoms, dtype=np.float32)
    degree = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / np.clip(degree, a_min=1.0, a_max=None)

    return MoleculeGraph(
        drug_id=drug_id,
        smiles=smiles,
        node_features=torch.from_numpy(node_features),
        adjacency=torch.from_numpy(adjacency),
    )


class MoleculeDataset(Dataset):
    def __init__(self, graphs: Sequence[MoleculeGraph]):
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> MoleculeGraph:
        return self.graphs[index]


class FingerprintDataset(Dataset):
    def __init__(self, drugs: Sequence[FingerprintDrug]):
        self.drugs = list(drugs)

    def __len__(self) -> int:
        return len(self.drugs)

    def __getitem__(self, index: int) -> FingerprintDrug:
        return self.drugs[index]


def collate_graphs(batch: Sequence[MoleculeGraph]) -> Dict[str, torch.Tensor]:
    max_nodes = max(graph.node_features.shape[0] for graph in batch)
    feature_dim = batch[0].node_features.shape[1]
    batch_size = len(batch)

    features = torch.zeros(batch_size, max_nodes, feature_dim, dtype=torch.float32)
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.float32)

    smiles_list = []
    drug_ids = []
    for idx, graph in enumerate(batch):
        num_nodes = graph.node_features.shape[0]
        features[idx, :num_nodes] = graph.node_features
        adjacency[idx, :num_nodes, :num_nodes] = graph.adjacency
        mask[idx, :num_nodes] = 1.0
        smiles_list.append(graph.smiles)
        drug_ids.append(graph.drug_id)

    return {
        "features": features,
        "adjacency": adjacency,
        "mask": mask,
        "smiles": smiles_list,
        "drug_ids": drug_ids,
    }


def collate_fingerprints(batch: Sequence[FingerprintDrug]) -> Dict[str, torch.Tensor]:
    fingerprints = torch.stack([drug.fingerprint for drug in batch], dim=0)
    drug_ids = [drug.drug_id for drug in batch]
    return {
        "fingerprints": fingerprints,
        "drug_ids": drug_ids,
    }


def _checkpoint_path_text(path: Optional[Path]) -> Optional[str]:
    return None if path is None else str(path.resolve())


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.bmm(adjacency, x)
        return self.linear(aggregated)


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder_gcn1 = GCNLayer(input_dim, hidden_dim)
        self.encoder_gcn2 = GCNLayer(hidden_dim, embedding_dim)
        self.node_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.encoder_gcn1(features, adjacency))
        node_embeddings = self.encoder_gcn2(hidden, adjacency)
        masked_embeddings = node_embeddings * mask.unsqueeze(-1)
        graph_embeddings = masked_embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return node_embeddings, graph_embeddings

    def decode(
        self,
        node_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reconstructed_features = self.node_decoder(node_embeddings)
        reconstructed_adjacency = torch.matmul(node_embeddings, node_embeddings.transpose(1, 2))
        return reconstructed_features, reconstructed_adjacency

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_embeddings, graph_embeddings = self.encode(features, adjacency, mask)
        reconstructed_features, reconstructed_adjacency = self.decode(node_embeddings)
        return reconstructed_features, reconstructed_adjacency, graph_embeddings


class FingerprintAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return self.encoder(fingerprints)

    def forward(self, fingerprints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(fingerprints)
        reconstructed = self.decoder(embeddings)
        return reconstructed, embeddings


def masked_feature_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = ((pred - target) ** 2).mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_adjacency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def train_autoencoder(
    model: GraphAutoencoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            features = batch["features"].to(device)
            adjacency = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            pred_features, pred_adjacency, _ = model(features, adjacency, mask)
            feature_loss = masked_feature_loss(pred_features, features, mask)
            adjacency_loss = masked_adjacency_loss(pred_adjacency, adjacency, mask)
            loss = feature_loss + adjacency_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch:03d}/{epochs:03d}  loss={mean_loss:.6f}")


def collect_embeddings(
    model: GraphAutoencoder,
    graphs: Sequence[MoleculeGraph],
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    loader = DataLoader(
        MoleculeDataset(graphs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    model.eval()

    embeddings: Dict[str, np.ndarray] = {}
    metadata: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            adjacency = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)

            _, graph_embeddings = model.encode(features, adjacency, mask)
            graph_embeddings_np = graph_embeddings.cpu().numpy().astype(np.float32)

            for idx, smiles in enumerate(batch["smiles"]):
                embeddings[smiles] = graph_embeddings_np[idx]
                metadata.append(
                    {
                        "drug_id": batch["drug_ids"][idx],
                        "smiles": smiles,
                        "embedding_index": len(metadata),
                    }
                )

    return embeddings, metadata


def save_embeddings(path: Path, embeddings: Dict[str, np.ndarray]) -> None:
    if path.suffix.lower() == ".npy":
        np.save(path, embeddings, allow_pickle=True)
        return
    with path.open("wb") as handle:
        pickle.dump(embeddings, handle)


def save_autoencoder_checkpoint(
    path: Path,
    model: nn.Module,
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    source_metadata: Dict[str, object],
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "model_type": model_type,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "embedding_dim": int(embedding_dim),
        "source_metadata": source_metadata,
    }
    torch.save(payload, path)


def load_autoencoder_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    payload = torch.load(path, map_location=device)
    model_type = str(payload.get("model_type", "")).strip().lower()
    if model_type == "graph":
        model = GraphAutoencoder(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            embedding_dim=int(payload["embedding_dim"]),
        )
    elif model_type == "fingerprint":
        model = FingerprintAutoencoder(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            embedding_dim=int(payload["embedding_dim"]),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model_type in {path}: {payload.get('model_type')}")

    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def read_smiles_text_file(path: Path) -> List[str]:
    smiles_list: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        smiles = normalize_text(line)
        if smiles is not None:
            smiles_list.append(smiles)
    return smiles_list


def collect_unique_inference_smiles(
    cli_smiles: Optional[Sequence[str]],
    smiles_file: Optional[Path],
) -> List[Tuple[str, str]]:
    unique_pairs: Dict[str, str] = {}

    if cli_smiles:
        for raw_smiles in cli_smiles:
            smiles = normalize_text(raw_smiles)
            if smiles is not None:
                unique_pairs.setdefault(smiles, smiles)

    if smiles_file is not None:
        for smiles in read_smiles_text_file(smiles_file):
            unique_pairs.setdefault(smiles, smiles)

    if not unique_pairs:
        raise ValueError("Inference mode requires at least one SMILES via --infer-smiles or --infer-smiles-file.")

    return sorted(unique_pairs.items(), key=lambda item: item[0])


def train_fingerprint_autoencoder(
    model: FingerprintAutoencoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            fingerprints = batch["fingerprints"].to(device)

            optimizer.zero_grad()
            reconstructed, _ = model(fingerprints)
            loss = criterion(reconstructed, fingerprints)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch:03d}/{epochs:03d}  loss={mean_loss:.6f}")


def collect_fingerprint_embeddings(
    model: FingerprintAutoencoder,
    drugs: Sequence[FingerprintDrug],
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    loader = DataLoader(
        FingerprintDataset(drugs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fingerprints,
    )
    model.eval()

    embeddings: Dict[str, np.ndarray] = {}
    metadata: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            fingerprints = batch["fingerprints"].to(device)
            embedding_vectors = model.encode(fingerprints).cpu().numpy().astype(np.float32)

            for idx, drug_id in enumerate(batch["drug_ids"]):
                embeddings[drug_id] = embedding_vectors[idx]
                metadata.append(
                    {
                        "drug_id": drug_id,
                        "smiles": None,
                        "embedding_index": len(metadata),
                    }
                )

    return embeddings, metadata


def run_inference(args: argparse.Namespace) -> None:
    if Chem is None:
        raise RuntimeError(
            "RDKit is required to convert inference SMILES into molecular graphs. Install rdkit before running this script."
        )

    if not args.checkpoint_input:
        raise ValueError("Inference mode requires --checkpoint-input.")

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint_input).expanduser().resolve()
    model, checkpoint_payload = load_autoencoder_checkpoint(checkpoint_path, device)
    if str(checkpoint_payload.get("model_type", "")).strip().lower() != "graph":
        raise ValueError(
            "Inference from SMILES is only supported for graph autoencoder checkpoints, "
            f"but {checkpoint_path} stores model_type={checkpoint_payload.get('model_type')!r}."
        )

    smiles_file = Path(args.infer_smiles_file).expanduser().resolve() if args.infer_smiles_file else None
    unique_smiles = collect_unique_inference_smiles(args.infer_smiles, smiles_file)
    print(f"Running inference for {len(unique_smiles)} SMILES.")

    graphs: List[MoleculeGraph] = []
    skipped: List[Tuple[str, str, str]] = []
    for drug_id, smiles in unique_smiles:
        try:
            graphs.append(smiles_to_graph(drug_id, smiles))
        except Exception as exc:
            skipped.append((drug_id, smiles, str(exc)))

    if not graphs:
        raise ValueError("No valid molecule graphs could be created for inference.")

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
        "embedding_source": "smiles_graph_autoencoder_inference",
        "checkpoint_input": str(checkpoint_path),
        "checkpoint_output": None,
        "num_requested_smiles": len(unique_smiles),
        "num_embedded_smiles": len(graphs),
        "num_skipped_smiles": len(skipped),
        "embedding_dim": int(checkpoint_payload["embedding_dim"]),
        "entries": metadata,
        "skipped": [
            {"drug_id": drug_id, "smiles": smiles, "reason": reason}
            for drug_id, smiles, reason in skipped
        ],
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))

    print(f"Saved inference embeddings to {output_path}")
    print(f"Saved inference metadata to {metadata_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.infer_smiles or args.infer_smiles_file:
        run_inference(args)
        return

    dataset_path = Path(args.input)
    smiles_dir = Path(args.smiles_dir).expanduser().resolve() if args.smiles_dir else None
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    checkpoint_output_path = Path(args.checkpoint_output)
    device = torch.device(args.device)

    if smiles_dir is not None:
        if Chem is None:
            raise RuntimeError(
                "RDKit is required to convert SMILES into molecular graphs. Install rdkit before running this script."
            )
        unique_drugs = read_smiles_dir(smiles_dir)
        print(f"Found {len(unique_drugs)} unique SMILES in {smiles_dir}.")

        graphs: List[MoleculeGraph] = []
        skipped: List[Tuple[str, str, str]] = []
        for drug_id, smiles in unique_drugs:
            try:
                graphs.append(smiles_to_graph(drug_id, smiles))
            except Exception as exc:
                skipped.append((drug_id, smiles, str(exc)))

        if not graphs:
            raise ValueError("No valid molecule graphs could be created from the SMILES folder.")

        if skipped:
            print(f"Skipped {len(skipped)} SMILES due to invalid or unparsable graphs.")

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
            "input_dataset": None,
            "input_smiles_dir": str(smiles_dir),
            "embedding_source": "smiles_graph_autoencoder",
            "checkpoint_output": _checkpoint_path_text(checkpoint_output_path),
            "num_unique_drugs": len(unique_drugs),
            "num_trained_graphs": len(graphs),
            "num_skipped_graphs": len(skipped),
            "embedding_dim": args.embedding_dim,
            "entries": metadata,
            "skipped": [
                {"drug_id": drug_id, "smiles": smiles, "reason": reason}
                for drug_id, smiles, reason in skipped
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
                "input_dataset": None,
                "input_smiles_dir": str(smiles_dir),
                "embedding_source": "smiles_graph_autoencoder",
            },
        )
    else:
        ensure_dependencies()
        dataset = load_pickle(dataset_path)
        try:
            unique_drugs = extract_unique_drugs(dataset)
        except ValueError as smiles_error:
            unique_fingerprint_drugs = extract_unique_drugs_from_fingerprints(dataset)
            print(
                "No SMILES were found in the dataset. Falling back to the dataset's precomputed "
                "drug fingerprints for embedding learning."
            )
            print(f"Found {len(unique_fingerprint_drugs)} unique drugs with fingerprint vectors.")

            train_loader = DataLoader(
                FingerprintDataset(unique_fingerprint_drugs),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fingerprints,
            )

            input_dim = unique_fingerprint_drugs[0].fingerprint.shape[0]
            model = FingerprintAutoencoder(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
            ).to(device)

            train_fingerprint_autoencoder(
                model=model,
                loader=train_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            embeddings, metadata = collect_fingerprint_embeddings(
                model=model,
                drugs=unique_fingerprint_drugs,
                device=device,
                batch_size=args.batch_size,
            )

            metadata_payload = {
                "input_dataset": str(dataset_path),
                "embedding_source": "fingerprint_fallback",
                "checkpoint_output": _checkpoint_path_text(checkpoint_output_path),
                "fallback_reason": str(smiles_error),
                "num_unique_drugs": len(unique_fingerprint_drugs),
                "embedding_dim": args.embedding_dim,
                "entries": metadata,
                "skipped": [],
            }
            save_autoencoder_checkpoint(
                path=checkpoint_output_path,
                model=model,
                model_type="fingerprint",
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                source_metadata={
                    "input_dataset": str(dataset_path),
                    "embedding_source": "fingerprint_fallback",
                    "fallback_reason": str(smiles_error),
                },
            )
        else:
            if Chem is None:
                raise RuntimeError(
                    "RDKit is required to convert SMILES into molecular graphs. Install rdkit before running this script."
                )

            print(f"Found {len(unique_drugs)} unique drugs with SMILES.")

            graphs: List[MoleculeGraph] = []
            skipped: List[Tuple[str, str, str]] = []
            for drug_id, smiles in unique_drugs:
                try:
                    graphs.append(smiles_to_graph(drug_id, smiles))
                except Exception as exc:
                    skipped.append((drug_id, smiles, str(exc)))

            if not graphs:
                raise ValueError("No valid molecule graphs could be created from the dataset SMILES.")

            if skipped:
                print(f"Skipped {len(skipped)} drugs due to invalid or unparsable SMILES.")

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
                "input_dataset": str(dataset_path),
                "embedding_source": "smiles_graph_autoencoder",
                "checkpoint_output": _checkpoint_path_text(checkpoint_output_path),
                "num_unique_drugs": len(unique_drugs),
                "num_trained_graphs": len(graphs),
                "num_skipped_graphs": len(skipped),
                "embedding_dim": args.embedding_dim,
                "entries": metadata,
                "skipped": [
                    {"drug_id": drug_id, "smiles": smiles, "reason": reason}
                    for drug_id, smiles, reason in skipped
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
                    "input_dataset": str(dataset_path),
                    "embedding_source": "smiles_graph_autoencoder",
                },
            )

    save_embeddings(output_path, embeddings)
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))

    print(f"Saved embeddings to {output_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved checkpoint to {checkpoint_output_path}")


if __name__ == "__main__":
    main()
