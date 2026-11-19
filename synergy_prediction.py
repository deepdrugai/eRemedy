#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_INPUT = "dataset_ONEIL.pkl"
DEFAULT_MOLECULE_EMBEDDINGS = "molecule_smiles_embeddings.pkl"
DEFAULT_FRAGMENT_EMBEDDINGS = "fragment_smiles_embeddings.pkl"
DEFAULT_FRAGMENT_MAP = "smiles_to_fragments.pkl"
DEFAULT_DRUG_SMILES_MAP = "everything_drug_smiles_map.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--molecule-embeddings", default=DEFAULT_MOLECULE_EMBEDDINGS)
    parser.add_argument("--fragment-embeddings", default=DEFAULT_FRAGMENT_EMBEDDINGS)
    parser.add_argument("--fragment-map", default=DEFAULT_FRAGMENT_MAP)
    parser.add_argument("--drug-smiles-map", default=DEFAULT_DRUG_SMILES_MAP)
    parser.add_argument("--outdir", default="oneil_synergy_runs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-sym", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--d-cell", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--include-no-fragments",
        action="store_true",
        help="Keep rows even when a drug has no usable fragments by using a zero fragment vector.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_embedding_pickle(path: Path) -> Dict[str, np.ndarray]:
    payload = load_pickle(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in embedding pickle: {path}")

    embeddings = {}
    for key, value in payload.items():
        name = normalize_text(key)
        if name is None:
            continue
        embeddings[name] = np.asarray(value, dtype=np.float32)
    if not embeddings:
        raise ValueError(f"No embeddings found in: {path}")
    return embeddings


def load_drug_smiles_map(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path)
    required = {"drug", "smiles"}
    if not required.issubset(df.columns):
        raise ValueError(f"Drug-to-SMILES CSV must contain columns {required}: {path}")

    mapping = {}
    for drug, smiles in zip(df["drug"], df["smiles"]):
        drug_name = normalize_text(drug)
        smiles_text = normalize_text(smiles)
        if drug_name is None or smiles_text is None:
            continue
        mapping.setdefault(drug_name, smiles_text)
    return mapping


def extract_target(meta: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    if "y_loewe" in meta.columns:
        target = pd.to_numeric(meta["y_loewe"], errors="coerce").to_numpy(dtype=np.float32)
    else:
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError("Expected dataset `y` array with at least 2 columns for loewe.")
        target = np.asarray(y[:, 1], dtype=np.float32)
    return target


class CellOmicsEncoderAttn(nn.Module):
    def __init__(self, in_ch: int = 6, d_cell: int = 256, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_cell),
        )
        self.attn = nn.Sequential(
            nn.Linear(d_cell, d_cell),
            nn.Tanh(),
            nn.Linear(d_cell, 1),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, cell: torch.Tensor, return_attn: bool = False):
        hidden = self.proj(cell)
        hidden = self.drop(hidden)
        logits = self.attn(hidden).squeeze(-1)
        alpha = torch.softmax(logits, dim=1)
        pooled = torch.einsum("bg,bgd->bd", alpha, hidden)
        if return_attn:
            return pooled, alpha
        return pooled


class FragmentAttentionPool(nn.Module):
    def __init__(self, d_frag: int, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_frag, d_frag),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_frag, 1),
        )

    def forward(self, fragments: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.attn(fragments).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        alpha = torch.softmax(logits, dim=1)
        alpha = alpha * mask.float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bf,bfd->bd", alpha, fragments)


class SynergyDataset(Dataset):
    def __init__(
        self,
        cells: np.ndarray,
        mol_a: np.ndarray,
        mol_b: np.ndarray,
        frag_a: Sequence[np.ndarray],
        frag_b: Sequence[np.ndarray],
        targets: np.ndarray,
    ):
        self.cells = cells
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.frag_a = list(frag_a)
        self.frag_b = list(frag_b)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            self.cells[index],
            self.mol_a[index],
            self.mol_b[index],
            self.frag_a[index],
            self.frag_b[index],
            self.targets[index],
        )


def collate_batch(batch):
    cells, mol_a, mol_b, frag_a, frag_b, targets = zip(*batch)

    cell_tensor = torch.from_numpy(np.stack(cells, axis=0).astype(np.float32))
    mol_a_tensor = torch.from_numpy(np.stack(mol_a, axis=0).astype(np.float32))
    mol_b_tensor = torch.from_numpy(np.stack(mol_b, axis=0).astype(np.float32))
    target_tensor = torch.from_numpy(np.asarray(targets, dtype=np.float32))

    frag_dim = frag_a[0].shape[1]
    max_frag_a = max(fragment.shape[0] for fragment in frag_a)
    max_frag_b = max(fragment.shape[0] for fragment in frag_b)

    frag_a_tensor = torch.zeros(len(batch), max_frag_a, frag_dim, dtype=torch.float32)
    frag_b_tensor = torch.zeros(len(batch), max_frag_b, frag_dim, dtype=torch.float32)
    frag_a_mask = torch.zeros(len(batch), max_frag_a, dtype=torch.bool)
    frag_b_mask = torch.zeros(len(batch), max_frag_b, dtype=torch.bool)

    for idx, fragment_array in enumerate(frag_a):
        n_frag = fragment_array.shape[0]
        frag_a_tensor[idx, :n_frag] = torch.from_numpy(fragment_array.astype(np.float32))
        frag_a_mask[idx, :n_frag] = True

    for idx, fragment_array in enumerate(frag_b):
        n_frag = fragment_array.shape[0]
        frag_b_tensor[idx, :n_frag] = torch.from_numpy(fragment_array.astype(np.float32))
        frag_b_mask[idx, :n_frag] = True

    return (
        cell_tensor,
        mol_a_tensor,
        mol_b_tensor,
        frag_a_tensor,
        frag_a_mask,
        frag_b_tensor,
        frag_b_mask,
        target_tensor,
    )


class SynergyModel(nn.Module):
    def __init__(
        self,
        d_graph: int,
        d_frag: int,
        d_cell: int = 256,
        hidden: int = 512,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.cell_enc = CellOmicsEncoderAttn(in_ch=6, d_cell=d_cell, hidden=d_cell, dropout=dropout)
        self.frag_pool = FragmentAttentionPool(d_frag=d_frag, dropout=dropout)

        self.d_drug = d_graph + d_frag
        self.head = nn.Sequential(
            nn.Linear(d_cell + 4 * self.d_drug, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def encode_drug(
        self,
        mol_embedding: torch.Tensor,
        frag_embeddings: torch.Tensor,
        frag_mask: torch.Tensor,
    ) -> torch.Tensor:
        z_graph = mol_embedding
        z_frag = self.frag_pool(frag_embeddings, frag_mask)
        return torch.cat([z_graph, z_frag], dim=-1)

    def forward(
        self,
        cell: torch.Tensor,
        mol_a: torch.Tensor,
        mol_b: torch.Tensor,
        frag_a: torch.Tensor,
        frag_a_mask: torch.Tensor,
        frag_b: torch.Tensor,
        frag_b_mask: torch.Tensor,
    ) -> torch.Tensor:
        z_cell = self.cell_enc(cell)
        z_a = self.encode_drug(mol_a, frag_a, frag_a_mask)
        z_b = self.encode_drug(mol_b, frag_b, frag_b_mask)
        z_pair = torch.cat([z_a, z_b, (z_a - z_b).abs(), z_a * z_b], dim=-1)
        z = torch.cat([z_pair, z_cell], dim=-1)
        return self.head(z).squeeze(-1)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if a.size < 2:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    if method == "spearman":
        ra = pd.Series(a).rank(method="average").to_numpy()
        rb = pd.Series(b).rank(method="average").to_numpy()
        return float(np.corrcoef(ra, rb)[0, 1])
    raise ValueError(method)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {
            "count": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
    yt = y_true[valid]
    yp = y_pred[valid]
    err = yp - yt
    mse = float((err * err).mean())
    return {
        "count": int(valid.sum()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.abs(err).mean()),
        "pearson": _safe_corrcoef(yt, yp, "pearson"),
        "spearman": _safe_corrcoef(yt, yp, "spearman"),
    }


def _collect_predictions(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for cell, mol_a, mol_b, frag_a, frag_a_mask, frag_b, frag_b_mask, y in loader:
            pred = model(
                cell.to(device),
                mol_a.to(device),
                mol_b.to(device),
                frag_a.to(device),
                frag_a_mask.to(device),
                frag_b.to(device),
                frag_b_mask.to(device),
            )
            preds.append(pred.cpu().numpy())
            truths.append(y.numpy())
    return np.concatenate(truths, axis=0), np.concatenate(preds, axis=0)


def evaluate(model: nn.Module, loader: DataLoader, device: str, return_arrays: bool = False):
    y_true, y_pred = _collect_predictions(model, loader, device)
    metrics = _metric_dict(y_true, y_pred)
    if return_arrays:
        return metrics, y_true, y_pred
    return metrics


def _train_epoch(model, loader, optimizer, args):
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_sym_loss = 0.0
    n = 0

    for cell, mol_a, mol_b, frag_a, frag_a_mask, frag_b, frag_b_mask, y in loader:
        cell = cell.to(args.device)
        mol_a = mol_a.to(args.device)
        mol_b = mol_b.to(args.device)
        frag_a = frag_a.to(args.device)
        frag_a_mask = frag_a_mask.to(args.device)
        frag_b = frag_b.to(args.device)
        frag_b_mask = frag_b_mask.to(args.device)
        y = y.to(args.device)

        pred_ab = model(cell, mol_a, mol_b, frag_a, frag_a_mask, frag_b, frag_b_mask)
        pred_loss = F.mse_loss(pred_ab, y)

        pred_ba = model(cell, mol_b, mol_a, frag_b, frag_b_mask, frag_a, frag_a_mask)
        sym_loss = F.mse_loss(pred_ab, pred_ba)

        loss = pred_loss + args.lambda_sym * sym_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch_size = y.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_pred_loss += float(pred_loss.item()) * batch_size
        total_sym_loss += float(sym_loss.item()) * batch_size
        n += batch_size

    return total_loss / max(n, 1), total_pred_loss / max(n, 1), total_sym_loss / max(n, 1)


def train_one(model, train_loader, val_loader, args):
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_rmse = float("inf")
    best_state = None
    history = {
        "epoch": [],
        "train_rmse": [],
        "train_pred_rmse": [],
        "train_sym_rmse": [],
        "train_mse": [],
        "val_rmse": [],
        "val_mse": [],
        "val_mae": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_mse, train_pred_loss, train_sym_loss = _train_epoch(model, train_loader, optimizer, args)
        val_metrics = evaluate(model, val_loader, args.device)
        train_rmse = float(np.sqrt(train_mse)) if np.isfinite(train_mse) else float("nan")
        train_pred_rmse = float(np.sqrt(train_pred_loss)) if np.isfinite(train_pred_loss) else float("nan")
        train_sym_rmse = float(np.sqrt(train_sym_loss)) if np.isfinite(train_sym_loss) else float("nan")

        history["epoch"].append(epoch)
        history["train_rmse"].append(train_rmse)
        history["train_pred_rmse"].append(train_pred_rmse)
        history["train_sym_rmse"].append(train_sym_rmse)
        history["train_mse"].append(float(train_mse))
        history["val_rmse"].append(float(val_metrics["rmse"]))
        history["val_mse"].append(float(val_metrics["mse"]))
        history["val_mae"].append(float(val_metrics["mae"]))

        print(
            f"Epoch {epoch:03d} | "
            f"train RMSE={train_rmse:.6f} | train MSE={train_mse:.6f} | "
            f"val RMSE={val_metrics['rmse']:.6f} | val MSE={val_metrics['mse']:.6f} | "
            f"val MAE={val_metrics['mae']:.6f} | val Pearson={val_metrics['pearson']:.6f} | "
            f"val Spearman={val_metrics['spearman']:.6f}"
        )

        if val_metrics["rmse"] < best_rmse:
            best_rmse = float(val_metrics["rmse"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def plot_rmse_curve(history, out_path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(history["epoch"], history["train_rmse"], marker="o", label="train_rmse")
    plt.plot(history["epoch"], history["val_rmse"], marker="o", label="val_rmse")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_true_vs_pred(y_true, y_pred, out_path):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[valid]
    yp = y_pred[valid]
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(yt, yp, s=8, alpha=0.45)
    if yt.size > 0:
        lo = float(np.nanmin([yt.min(), yp.min()]))
        hi = float(np.nanmax([yt.max(), yp.max()]))
        plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_checkpoint(model, out_path, meta_dict):
    torch.save({"state_dict": model.state_dict(), **meta_dict}, out_path)


def dataset_summary(meta: pd.DataFrame, n_total: int, n_valid: int):
    unique_drugs = []
    if "drug_a" in meta.columns and "drug_b" in meta.columns:
        unique_drugs = sorted(set(meta["drug_a"].dropna().astype(str)).union(set(meta["drug_b"].dropna().astype(str))))
    unique_cells = sorted(meta["cell_line_name"].dropna().astype(str).unique().tolist()) if "cell_line_name" in meta.columns else []
    return {
        "n_total": int(n_total),
        "n_valid": int(n_valid),
        "n_unique_cell_lines": int(len(unique_cells)),
        "n_unique_drugs_total": int(len(unique_drugs)),
        "n_unique_drug_a": int(meta["drug_a"].nunique()) if "drug_a" in meta.columns else None,
        "n_unique_drug_b": int(meta["drug_b"].nunique()) if "drug_b" in meta.columns else None,
        "cell_line_names": unique_cells,
    }


def _split_indices(n: int, val_fraction: float, test_fraction: float, seed: int):
    if n <= 2:
        raise ValueError("Need at least 3 samples to build train/val/test splits.")
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_test + n_val >= n:
        n_test = max(1, n // 10)
        n_val = max(1, n // 10)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    if len(train_idx) == 0:
        raise ValueError("Split fractions leave no training samples.")
    return train_idx, val_idx, test_idx


def _to_fragment_matrix(fragment_smiles: List[str], fragment_embeddings: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    vectors = []
    for smiles in fragment_smiles:
        if smiles in fragment_embeddings:
            vectors.append(fragment_embeddings[smiles])
    if not vectors:
        return None
    return np.stack(vectors, axis=0).astype(np.float32)


def _build_examples(
    dataset,
    molecule_embeddings: Dict[str, np.ndarray],
    fragment_embeddings: Dict[str, np.ndarray],
    fragment_map: Dict[str, List[str]],
    drug_smiles_map: Dict[str, str],
    include_no_fragments: bool,
):
    if not isinstance(dataset, dict):
        raise ValueError("Expected dataset pickle to contain a dict payload.")

    meta = dataset.get("meta")
    cell_omics = dataset.get("cell_omics")
    y = dataset.get("y")
    if meta is None or cell_omics is None or y is None:
        raise ValueError("Dataset must contain meta, cell_omics, and y.")
    if not isinstance(meta, pd.DataFrame):
        raise ValueError("Dataset `meta` must be a pandas DataFrame.")
    if "drug_a" not in meta.columns or "drug_b" not in meta.columns:
        raise ValueError("Dataset meta must contain drug_a and drug_b.")

    targets = extract_target(meta, np.asarray(y))
    cell_array = np.asarray(cell_omics, dtype=np.float32)
    if len(meta) != cell_array.shape[0] or len(meta) != len(targets):
        raise ValueError("meta, cell_omics, and target lengths do not align.")
    fragment_dim = next(iter(fragment_embeddings.values())).shape[0]

    cells = []
    mol_a = []
    mol_b = []
    frag_a = []
    frag_b = []
    scores = []
    meta_rows = []
    skipped = {
        "drug_a_smiles": 0,
        "drug_b_smiles": 0,
        "drug_a_embedding": 0,
        "drug_b_embedding": 0,
        "drug_a_fragments": 0,
        "drug_b_fragments": 0,
        "target": 0,
    }

    for idx, row in enumerate(meta.itertuples(index=False)):
        row_dict = row._asdict() if hasattr(row, "_asdict") else {
            meta.columns[col_idx]: row[col_idx] for col_idx in range(len(meta.columns))
        }
        drug_a_name = normalize_text(row_dict.get("drug_a"))
        drug_b_name = normalize_text(row_dict.get("drug_b"))
        if drug_a_name is None or drug_b_name is None:
            continue

        score = targets[idx]
        if not np.isfinite(score):
            skipped["target"] += 1
            continue

        smiles_a = drug_smiles_map.get(drug_a_name)
        smiles_b = drug_smiles_map.get(drug_b_name)
        if smiles_a is None:
            skipped["drug_a_smiles"] += 1
            continue
        if smiles_b is None:
            skipped["drug_b_smiles"] += 1
            continue

        z_graph_a = molecule_embeddings.get(smiles_a)
        z_graph_b = molecule_embeddings.get(smiles_b)
        if z_graph_a is None:
            skipped["drug_a_embedding"] += 1
            continue
        if z_graph_b is None:
            skipped["drug_b_embedding"] += 1
            continue

        fragment_list_a = fragment_map.get(smiles_a, [])
        fragment_list_b = fragment_map.get(smiles_b, [])
        z_frag_a = _to_fragment_matrix(fragment_list_a, fragment_embeddings)
        z_frag_b = _to_fragment_matrix(fragment_list_b, fragment_embeddings)
        if z_frag_a is None:
            if include_no_fragments:
                z_frag_a = np.zeros((1, fragment_dim), dtype=np.float32)
            else:
                skipped["drug_a_fragments"] += 1
                continue
        if z_frag_b is None:
            if include_no_fragments:
                z_frag_b = np.zeros((1, fragment_dim), dtype=np.float32)
            else:
                skipped["drug_b_fragments"] += 1
                continue

        cells.append(cell_array[idx])
        mol_a.append(np.asarray(z_graph_a, dtype=np.float32))
        mol_b.append(np.asarray(z_graph_b, dtype=np.float32))
        frag_a.append(z_frag_a)
        frag_b.append(z_frag_b)
        scores.append(float(score))
        meta_rows.append(
            {
                "drug_a": drug_a_name,
                "drug_b": drug_b_name,
                "drug_a_smiles": smiles_a,
                "drug_b_smiles": smiles_b,
                "cell_line_name": row_dict.get("cell_line_name"),
                "study_name": row_dict.get("study_name"),
                "y_loewe": float(score),
            }
        )

    if not meta_rows:
        raise ValueError(f"No usable rows after feature resolution. skipped={skipped}")

    return {
        "cells": np.stack(cells, axis=0).astype(np.float32),
        "mol_a": np.stack(mol_a, axis=0).astype(np.float32),
        "mol_b": np.stack(mol_b, axis=0).astype(np.float32),
        "frag_a": frag_a,
        "frag_b": frag_b,
        "targets": np.asarray(scores, dtype=np.float32),
        "meta": pd.DataFrame(meta_rows),
        "total_rows": int(len(meta)),
        "valid_rows": int(len(meta_rows)),
        "skipped": skipped,
    }


def _slice_examples(examples, indices: np.ndarray):
    return (
        examples["cells"][indices],
        examples["mol_a"][indices],
        examples["mol_b"][indices],
        [examples["frag_a"][idx] for idx in indices.tolist()],
        [examples["frag_b"][idx] for idx in indices.tolist()],
        examples["targets"][indices],
        examples["meta"].iloc[indices].reset_index(drop=True),
        int(examples["total_rows"] if len(indices) == examples["valid_rows"] else len(indices)),
        int(len(indices)),
        examples["skipped"] if len(indices) == examples["valid_rows"] else {},
    )


def _build_split_bundle(args):
    dataset = load_pickle(Path(args.input))
    molecule_embeddings = load_embedding_pickle(Path(args.molecule_embeddings))
    fragment_embeddings = load_embedding_pickle(Path(args.fragment_embeddings))
    fragment_map = load_pickle(Path(args.fragment_map))
    drug_smiles_map = load_drug_smiles_map(Path(args.drug_smiles_map))

    examples = _build_examples(
        dataset=dataset,
        molecule_embeddings=molecule_embeddings,
        fragment_embeddings=fragment_embeddings,
        fragment_map=fragment_map,
        drug_smiles_map=drug_smiles_map,
        include_no_fragments=args.include_no_fragments,
    )

    train_idx, val_idx, test_idx = _split_indices(
        n=examples["valid_rows"],
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    return {
        "train": _slice_examples(examples, train_idx),
        "val": _slice_examples(examples, val_idx),
        "test": _slice_examples(examples, test_idx),
        "dataset_total": examples["total_rows"],
        "dataset_valid": examples["valid_rows"],
        "dataset_skipped": examples["skipped"],
        "molecule_dim": int(examples["mol_a"].shape[1]),
        "fragment_dim": int(examples["frag_a"][0].shape[1]),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    data = _build_split_bundle(args)

    tr_cell, tr_mol_a, tr_mol_b, tr_frag_a, tr_frag_b, tr_y, tr_meta, tr_total, tr_valid, tr_skip = data["train"]
    va_cell, va_mol_a, va_mol_b, va_frag_a, va_frag_b, va_y, va_meta, va_total, va_valid, va_skip = data["val"]
    te_cell, te_mol_a, te_mol_b, te_frag_a, te_frag_b, te_y, te_meta, te_total, te_valid, te_skip = data["test"]

    print("dataset total rows:", data["dataset_total"])
    print("dataset usable rows:", data["dataset_valid"])
    print("dataset skipped:", data["dataset_skipped"])
    print("train split:", tr_valid)
    print("val split:", va_valid)
    print("test split:", te_valid)

    train_loader = DataLoader(
        SynergyDataset(tr_cell, tr_mol_a, tr_mol_b, tr_frag_a, tr_frag_b, tr_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        SynergyDataset(va_cell, va_mol_a, va_mol_b, va_frag_a, va_frag_b, va_y),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        SynergyDataset(te_cell, te_mol_a, te_mol_b, te_frag_a, te_frag_b, te_y),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_batch,
    )

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    full_meta = pd.concat([tr_meta, va_meta, te_meta], axis=0, ignore_index=True)
    ds_summary = dataset_summary(full_meta, data["dataset_total"], data["dataset_valid"])
    ds_summary.update(
        {
            "input_dataset": str(Path(args.input).resolve()),
            "molecule_embeddings": str(Path(args.molecule_embeddings).resolve()),
            "fragment_embeddings": str(Path(args.fragment_embeddings).resolve()),
            "fragment_map": str(Path(args.fragment_map).resolve()),
            "drug_smiles_map": str(Path(args.drug_smiles_map).resolve()),
            "target": "loewe",
            "seed": args.seed,
            "train_valid": tr_valid,
            "val_valid": va_valid,
            "test_valid": te_valid,
            "dataset_skipped": data["dataset_skipped"],
            "molecule_dim": data["molecule_dim"],
            "fragment_dim": data["fragment_dim"],
            "include_no_fragments": args.include_no_fragments,
        }
    )
    (root / "dataset_summary.json").write_text(json.dumps(ds_summary, indent=2))

    with (root / "split_counts.csv").open("w", encoding="utf-8") as handle:
        handle.write("split,total,used\n")
        handle.write(f"train,{tr_total},{tr_valid}\n")
        handle.write(f"val,{va_total},{va_valid}\n")
        handle.write(f"test,{te_total},{te_valid}\n")

    model = SynergyModel(
        d_graph=data["molecule_dim"],
        d_frag=data["fragment_dim"],
        d_cell=args.d_cell,
        hidden=args.hidden,
        dropout=args.dropout,
    )
    model, history = train_one(model, train_loader, val_loader, args)

    metrics_train, y_train_true, y_train_pred = evaluate(model, train_loader, args.device, return_arrays=True)
    metrics_val, y_val_true, y_val_pred = evaluate(model, val_loader, args.device, return_arrays=True)
    metrics_test, y_test_true, y_test_pred = evaluate(model, test_loader, args.device, return_arrays=True)

    plot_rmse_curve(history, root / "rmse_curve.png")
    plot_true_vs_pred(y_train_true, y_train_pred, root / "true_vs_pred_train.png")
    plot_true_vs_pred(y_val_true, y_val_pred, root / "true_vs_pred_val.png")
    plot_true_vs_pred(y_test_true, y_test_pred, root / "true_vs_pred_test.png")

    pd.DataFrame(history).to_csv(root / "rmse_history.csv", index=False)
    pd.DataFrame({"y_true": y_train_true, "y_pred": y_train_pred}).to_csv(root / "train_predictions.csv", index=False)
    pd.DataFrame({"y_true": y_val_true, "y_pred": y_val_pred}).to_csv(root / "val_predictions.csv", index=False)
    pd.DataFrame({"y_true": y_test_true, "y_pred": y_test_pred}).to_csv(root / "test_predictions.csv", index=False)

    metrics_payload = {
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
    }
    (root / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (root / "train_metrics.json").write_text(json.dumps(metrics_train, indent=2))
    (root / "val_metrics.json").write_text(json.dumps(metrics_val, indent=2))
    (root / "test_metrics.json").write_text(json.dumps(metrics_test, indent=2))

    with (root / "metrics_split.csv").open("w", encoding="utf-8") as handle:
        handle.write("split,mse,rmse,mae,pearson,spearman,count\n")
        handle.write(
            "train,{mse},{rmse},{mae},{pearson},{spearman},{count}\n".format(**metrics_train)
        )
        handle.write(
            "val,{mse},{rmse},{mae},{pearson},{spearman},{count}\n".format(**metrics_val)
        )
        handle.write(
            "test,{mse},{rmse},{mae},{pearson},{spearman},{count}\n".format(**metrics_test)
        )

    save_checkpoint(
        model,
        root / "model.pt",
        {
            "input_dataset": str(Path(args.input).resolve()),
            "target": "loewe",
            "molecule_dim": data["molecule_dim"],
            "fragment_dim": data["fragment_dim"],
            "d_cell": args.d_cell,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "seed": args.seed,
            "include_no_fragments": args.include_no_fragments,
        },
    )

    summary = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"train RMSE={metrics_train['rmse']:.6f}, train MSE={metrics_train['mse']:.6f}, "
        f"train MAE={metrics_train['mae']:.6f}"
    )
    print(
        f"val RMSE={metrics_val['rmse']:.6f}, val MSE={metrics_val['mse']:.6f}, "
        f"val MAE={metrics_val['mae']:.6f}, val Pearson={metrics_val['pearson']:.6f}, "
        f"val Spearman={metrics_val['spearman']:.6f}"
    )
    print(
        f"test RMSE={metrics_test['rmse']:.6f}, test MSE={metrics_test['mse']:.6f}, "
        f"test MAE={metrics_test['mae']:.6f}, test Pearson={metrics_test['pearson']:.6f}, "
        f"test Spearman={metrics_test['spearman']:.6f}"
    )


if __name__ == "__main__":
    main()
