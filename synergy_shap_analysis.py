#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors

from synergy_prediction import (
    DEFAULT_DRUG_SMILES_MAP,
    DEFAULT_FRAGMENT_EMBEDDINGS,
    DEFAULT_FRAGMENT_MAP,
    DEFAULT_MOLECULE_EMBEDDINGS,
    SynergyModel,
    _build_examples,
    load_drug_smiles_map,
    load_embedding_pickle,
    load_pickle,
    normalize_text,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explain a trained synergy model for all usable drug pairs in a given cell line. "
            "Outputs cell-line-wide modality SHAP summaries and deep fragment diagnostics for the top-ranked unique pairs."
        )
    )
    parser.add_argument("--checkpoint", default="oneil_synergy_runs/model.pt")
    parser.add_argument("--dataset", default=None, help="Defaults to the checkpoint's recorded input_dataset.")
    parser.add_argument("--molecule-embeddings", default=DEFAULT_MOLECULE_EMBEDDINGS)
    parser.add_argument("--fragment-embeddings", default=DEFAULT_FRAGMENT_EMBEDDINGS)
    parser.add_argument("--fragment-map", default=DEFAULT_FRAGMENT_MAP)
    parser.add_argument("--drug-smiles-map", default=DEFAULT_DRUG_SMILES_MAP)
    parser.add_argument("--cell-line", required=True)
    parser.add_argument("--drug-a", default=None)
    parser.add_argument("--drug-b", default=None)
    parser.add_argument(
        "--select-by",
        choices=["predicted_top", "observed_top", "index"],
        default="predicted_top",
        help="How to pick one example if multiple rows match the requested cell line.",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Index within the filtered cell-line rows when --select-by index is used.",
    )
    parser.add_argument(
        "--pair-order-sensitive",
        action="store_true",
        help="If set, drug_a and drug_b must match in the provided order.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-exact-fragments",
        type=int,
        default=10,
        help="Run exact fragment SHAP only when a drug has at most this many fragments.",
    )
    parser.add_argument(
        "--fragment-dim-shap-samples",
        type=int,
        default=512,
        help="Number of SHAP samples for within-fragment embedding-dimension attribution for each top-ranked pair.",
    )
    parser.add_argument(
        "--top-fragment-descriptors",
        type=int,
        default=5,
        help="Number of top fragments per selected drug to export with chemistry descriptors.",
    )
    parser.add_argument(
        "--fragment-detail-top-pairs",
        type=int,
        default=100,
        help="Number of top unique drug pairs in the cell line to run deep fragment-dimension SHAP on. Increase to 200 when you need a larger symbolic-regression seed pool.",
    )
    parser.add_argument(
        "--harmful-fragment-top-n",
        type=int,
        default=50,
        help="Number of strictly harmful fragments to export across the analyzed top pairs.",
    )
    parser.add_argument("--outdir", default="oneil_synergy_shap")
    return parser.parse_args()


class ExampleRecord:
    def __init__(
        self,
        cell: np.ndarray,
        mol_a: np.ndarray,
        mol_b: np.ndarray,
        frag_a: np.ndarray,
        frag_b: np.ndarray,
        target: float,
        meta_row: pd.Series,
    ):
        self.cell = cell
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.frag_a = frag_a
        self.frag_b = frag_b
        self.target = target
        self.meta_row = meta_row


MODALITY_NAMES = ["cell", "drug_a_graph", "drug_a_fragments", "drug_b_graph", "drug_b_fragments"]


def load_checkpoint(path: Path, device: str) -> Dict[str, object]:
    payload = torch.load(path, map_location=device)
    if "state_dict" not in payload:
        raise ValueError(f"Checkpoint does not contain state_dict: {path}")
    return payload


def build_model(checkpoint: Dict[str, object], device: str) -> SynergyModel:
    model = SynergyModel(
        d_graph=int(checkpoint["molecule_dim"]),
        d_frag=int(checkpoint["fragment_dim"]),
        d_cell=int(checkpoint["d_cell"]),
        hidden=int(checkpoint["hidden"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def resolve_dataset_path(cli_path: Optional[str], checkpoint: Dict[str, object]) -> Path:
    if cli_path:
        return Path(cli_path)
    ckpt_path = checkpoint.get("input_dataset")
    if not ckpt_path:
        raise ValueError("Dataset path was not provided and is missing from the checkpoint metadata.")
    return Path(str(ckpt_path))


def build_examples_bundle(args: argparse.Namespace, checkpoint: Dict[str, object]) -> Dict[str, object]:
    dataset_path = resolve_dataset_path(args.dataset, checkpoint)
    dataset = load_pickle(dataset_path)
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
        include_no_fragments=bool(checkpoint.get("include_no_fragments", False)),
    )
    examples["fragment_embeddings"] = fragment_embeddings
    examples["fragment_map"] = fragment_map
    return examples


def predict_batch(
    model: SynergyModel,
    cells: np.ndarray,
    mol_a: np.ndarray,
    mol_b: np.ndarray,
    frag_a: Sequence[np.ndarray],
    frag_b: Sequence[np.ndarray],
    device: str,
) -> np.ndarray:
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for idx in range(len(cells)):
            pred = model(
                torch.from_numpy(cells[idx : idx + 1]).to(device),
                torch.from_numpy(mol_a[idx : idx + 1]).to(device),
                torch.from_numpy(mol_b[idx : idx + 1]).to(device),
                torch.from_numpy(frag_a[idx][None, ...]).to(device),
                torch.ones((1, frag_a[idx].shape[0]), dtype=torch.bool, device=device),
                torch.from_numpy(frag_b[idx][None, ...]).to(device),
                torch.ones((1, frag_b[idx].shape[0]), dtype=torch.bool, device=device),
            )
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def filter_examples(examples: Dict[str, object], args: argparse.Namespace, model: SynergyModel) -> pd.DataFrame:
    meta = examples["meta"].copy()
    meta["row_id"] = np.arange(len(meta))
    target_cell = normalize_text(args.cell_line)
    if target_cell is None:
        raise ValueError("Cell line must be non-empty.")
    keep = meta["cell_line_name"].astype(str).str.strip().str.lower() == target_cell.lower()

    drug_a_query = normalize_text(args.drug_a)
    drug_b_query = normalize_text(args.drug_b)
    if drug_a_query and drug_b_query:
        if args.pair_order_sensitive:
            keep &= (
                meta["drug_a"].astype(str).str.strip().str.lower().eq(drug_a_query.lower())
                & meta["drug_b"].astype(str).str.strip().str.lower().eq(drug_b_query.lower())
            )
        else:
            pair_match = (
                meta["drug_a"].astype(str).str.strip().str.lower().eq(drug_a_query.lower())
                & meta["drug_b"].astype(str).str.strip().str.lower().eq(drug_b_query.lower())
            ) | (
                meta["drug_a"].astype(str).str.strip().str.lower().eq(drug_b_query.lower())
                & meta["drug_b"].astype(str).str.strip().str.lower().eq(drug_a_query.lower())
            )
            keep &= pair_match

    filtered = meta.loc[keep].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No usable rows matched the requested cell line / drug filters.")

    row_ids = filtered["row_id"].to_numpy(dtype=np.int64)
    filtered["predicted_synergy"] = predict_batch(
        model=model,
        cells=examples["cells"][row_ids],
        mol_a=examples["mol_a"][row_ids],
        mol_b=examples["mol_b"][row_ids],
        frag_a=[examples["frag_a"][i] for i in row_ids.tolist()],
        frag_b=[examples["frag_b"][i] for i in row_ids.tolist()],
        device=args.device,
    )

    if args.select_by == "predicted_top":
        filtered = filtered.sort_values("predicted_synergy", ascending=False).reset_index(drop=True)
    elif args.select_by == "observed_top":
        filtered = filtered.sort_values("y_loewe", ascending=False).reset_index(drop=True)

    return filtered


def select_example(examples: Dict[str, object], filtered: pd.DataFrame, args: argparse.Namespace) -> ExampleRecord:
    if args.select_by == "index":
        if args.example_index < 0 or args.example_index >= len(filtered):
            raise IndexError(f"--example-index must be between 0 and {len(filtered) - 1}")
        chosen = filtered.iloc[args.example_index]
    else:
        chosen = filtered.iloc[0]

    row_id = int(chosen["row_id"])
    return ExampleRecord(
        cell=examples["cells"][row_id],
        mol_a=examples["mol_a"][row_id],
        mol_b=examples["mol_b"][row_id],
        frag_a=examples["frag_a"][row_id],
        frag_b=examples["frag_b"][row_id],
        target=float(examples["targets"][row_id]),
        meta_row=chosen,
    )


def _torch_bool_mask(length: int, active: Optional[np.ndarray], device: str) -> torch.Tensor:
    if active is None:
        mask = np.ones(length, dtype=bool)
    else:
        mask = np.asarray(active, dtype=bool)
        if mask.shape != (length,):
            raise ValueError(f"Expected mask of shape {(length,)}, got {mask.shape}")
    return torch.from_numpy(mask[None, :]).to(device)


def predict_single_with_gates(
    model: SynergyModel,
    example: ExampleRecord,
    modality_gates: Sequence[float],
    device: str,
    fragment_presence_a: Optional[np.ndarray] = None,
    fragment_presence_b: Optional[np.ndarray] = None,
) -> float:
    gates = np.asarray(modality_gates, dtype=np.float32)
    if gates.shape != (5,):
        raise ValueError("modality_gates must have shape (5,)")

    cell = torch.from_numpy(example.cell[None, ...]).to(device) * float(gates[0])
    mol_a = torch.from_numpy(example.mol_a[None, ...]).to(device) * float(gates[1])
    mol_b = torch.from_numpy(example.mol_b[None, ...]).to(device) * float(gates[3])

    frag_a = np.asarray(example.frag_a, dtype=np.float32).copy()
    frag_b = np.asarray(example.frag_b, dtype=np.float32).copy()
    if fragment_presence_a is not None:
        frag_a *= np.asarray(fragment_presence_a, dtype=np.float32)[:, None]
    if fragment_presence_b is not None:
        frag_b *= np.asarray(fragment_presence_b, dtype=np.float32)[:, None]
    frag_a_tensor = torch.from_numpy(frag_a[None, ...]).to(device) * float(gates[2])
    frag_b_tensor = torch.from_numpy(frag_b[None, ...]).to(device) * float(gates[4])

    frag_a_mask = _torch_bool_mask(example.frag_a.shape[0], fragment_presence_a, device)
    frag_b_mask = _torch_bool_mask(example.frag_b.shape[0], fragment_presence_b, device)

    with torch.no_grad():
        pred = model(cell, mol_a, mol_b, frag_a_tensor, frag_a_mask, frag_b_tensor, frag_b_mask)
    return float(pred.item())


def exact_shap_values(
    n_features: int,
    value_fn,
    feature_names: Sequence[str],
) -> Dict[str, object]:
    factorial = math.factorial
    total_fact = factorial(n_features)
    contributions = np.zeros(n_features, dtype=np.float64)
    cache: Dict[Tuple[int, ...], float] = {}

    def subset_value(indices: Tuple[int, ...]) -> float:
        if indices not in cache:
            mask = np.zeros(n_features, dtype=np.float32)
            if indices:
                mask[list(indices)] = 1.0
            cache[indices] = float(value_fn(mask))
        return cache[indices]

    for feature_idx in range(n_features):
        other_features = [idx for idx in range(n_features) if idx != feature_idx]
        for subset_bits in range(1 << (n_features - 1)):
            subset = tuple(
                other_features[pos] for pos in range(n_features - 1) if (subset_bits >> pos) & 1
            )
            subset_with_feature = tuple(sorted(subset + (feature_idx,)))
            weight = (
                factorial(len(subset))
                * factorial(n_features - len(subset) - 1)
                / total_fact
            )
            contributions[feature_idx] += weight * (
                subset_value(subset_with_feature) - subset_value(subset)
            )

    baseline = subset_value(tuple())
    full_value = subset_value(tuple(range(n_features)))
    return {
        "feature_names": list(feature_names),
        "baseline_value": float(baseline),
        "full_value": float(full_value),
        "shap_values": contributions.astype(np.float64).tolist(),
    }


def fragment_attention_weights(
    frag_pool: nn.Module,
    fragment_array: np.ndarray,
    device: str,
) -> np.ndarray:
    fragments = torch.from_numpy(fragment_array[None, ...]).to(device)
    mask = torch.ones((1, fragment_array.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits = frag_pool.attn(fragments).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        alpha = torch.softmax(logits, dim=1)
        alpha = alpha * mask.float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return alpha.squeeze(0).detach().cpu().numpy()


def leave_one_out_fragment_effects(
    model: SynergyModel,
    example: ExampleRecord,
    base_prediction: float,
    device: str,
    which_drug: str,
) -> List[float]:
    if which_drug == "a":
        n_frag = example.frag_a.shape[0]
        values = []
        for idx in range(n_frag):
            mask = np.ones(n_frag, dtype=np.float32)
            mask[idx] = 0.0
            pred = predict_single_with_gates(
                model,
                example,
                modality_gates=np.ones(5, dtype=np.float32),
                fragment_presence_a=mask,
                device=device,
            )
            values.append(base_prediction - pred)
        return values

    if which_drug == "b":
        n_frag = example.frag_b.shape[0]
        values = []
        for idx in range(n_frag):
            mask = np.ones(n_frag, dtype=np.float32)
            mask[idx] = 0.0
            pred = predict_single_with_gates(
                model,
                example,
                modality_gates=np.ones(5, dtype=np.float32),
                fragment_presence_b=mask,
                device=device,
            )
            values.append(base_prediction - pred)
        return values

    raise ValueError(which_drug)


def fragment_shap(
    model: SynergyModel,
    example: ExampleRecord,
    device: str,
    which_drug: str,
) -> Dict[str, object]:
    if which_drug == "a":
        n_frag = example.frag_a.shape[0]

        def value_fn(mask: np.ndarray) -> float:
            return predict_single_with_gates(
                model,
                example,
                modality_gates=np.ones(5, dtype=np.float32),
                device=device,
                fragment_presence_a=mask,
            )

    elif which_drug == "b":
        n_frag = example.frag_b.shape[0]

        def value_fn(mask: np.ndarray) -> float:
            return predict_single_with_gates(
                model,
                example,
                modality_gates=np.ones(5, dtype=np.float32),
                device=device,
                fragment_presence_b=mask,
            )

    else:
        raise ValueError(which_drug)

    feature_names = [f"{which_drug}_fragment_{idx}" for idx in range(n_frag)]
    return exact_shap_values(n_frag, value_fn, feature_names)


def build_fragment_rows(
    smiles: str,
    fragment_map: Dict[str, List[str]],
    fragment_embeddings: Dict[str, np.ndarray],
    attention: np.ndarray,
    loo_effects: Sequence[float],
    shap_payload: Optional[Dict[str, object]],
) -> List[Dict[str, object]]:
    fragment_smiles = [
        frag_smiles
        for frag_smiles in fragment_map.get(smiles, [])
        if frag_smiles in fragment_embeddings
    ]
    rows = []
    shap_values = None
    if shap_payload is not None:
        shap_values = shap_payload["shap_values"]

    for idx in range(len(attention)):
        rows.append(
            {
                "fragment_index": idx,
                "fragment_smiles": fragment_smiles[idx] if idx < len(fragment_smiles) else None,
                "attention_weight": float(attention[idx]),
                "leave_one_out_delta": float(loo_effects[idx]),
                "fragment_shap_value": None if shap_values is None else float(shap_values[idx]),
            }
        )
    return rows


def summarize_modality_shap(shap_payload: Dict[str, object]) -> Dict[str, float]:
    values = shap_payload["shap_values"]
    mapping = {name: float(value) for name, value in zip(shap_payload["feature_names"], values)}
    return {
        "cell": mapping["cell"],
        "graph_total": mapping["drug_a_graph"] + mapping["drug_b_graph"],
        "fragment_total": mapping["drug_a_fragments"] + mapping["drug_b_fragments"],
    }


def fragment_descriptor_dict(smiles: str) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "rot_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


def pooled_fragment_vector(model: SynergyModel, fragment_array: np.ndarray, device: str) -> np.ndarray:
    fragments = torch.from_numpy(fragment_array[None, ...]).to(device)
    mask = torch.ones((1, fragment_array.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        pooled = model.frag_pool(fragments, mask)
    return pooled.squeeze(0).detach().cpu().numpy()


def modality_feature_values(model: SynergyModel, example: ExampleRecord, device: str) -> Dict[str, float]:
    pooled_a = pooled_fragment_vector(model, example.frag_a, device)
    pooled_b = pooled_fragment_vector(model, example.frag_b, device)
    return {
        "cell": float(np.linalg.norm(example.cell.reshape(-1), ord=2)),
        "drug_a_graph": float(np.linalg.norm(example.mol_a, ord=2)),
        "drug_a_fragments": float(np.linalg.norm(pooled_a, ord=2)),
        "drug_b_graph": float(np.linalg.norm(example.mol_b, ord=2)),
        "drug_b_fragments": float(np.linalg.norm(pooled_b, ord=2)),
    }


def compute_cell_line_modality_shap(
    model: SynergyModel,
    examples: Dict[str, object],
    filtered: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for filtered_idx in range(len(filtered)):
        selected = filtered.iloc[filtered_idx]
        example = select_example(
            examples=examples,
            filtered=filtered,
            args=argparse.Namespace(select_by="index", example_index=filtered_idx),
        )
        modality_shap = exact_shap_values(
            n_features=len(MODALITY_NAMES),
            feature_names=MODALITY_NAMES,
            value_fn=lambda gates, ex=example: predict_single_with_gates(model, ex, gates, device),
        )
        feature_values = modality_feature_values(model, example, device)
        record = {
            "row_id": int(selected["row_id"]),
            "cell_line_name": selected["cell_line_name"],
            "drug_a": selected["drug_a"],
            "drug_b": selected["drug_b"],
            "drug_a_smiles": selected["drug_a_smiles"],
            "drug_b_smiles": selected["drug_b_smiles"],
            "observed_synergy": float(example.target),
            "predicted_synergy": float(selected["predicted_synergy"]),
            "baseline_value": float(modality_shap["baseline_value"]),
            "full_value": float(modality_shap["full_value"]),
        }
        for name, shap_value in zip(modality_shap["feature_names"], modality_shap["shap_values"]):
            record[f"{name}_shap"] = float(shap_value)
            record[f"{name}_value"] = float(feature_values[name])
        record["graph_total_shap"] = record["drug_a_graph_shap"] + record["drug_b_graph_shap"]
        record["fragment_total_shap"] = record["drug_a_fragments_shap"] + record["drug_b_fragments_shap"]
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_cell_line_modality_shap(modality_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODALITY_NAMES + ["graph_total", "fragment_total"]:
        col = f"{name}_shap"
        if col not in modality_df.columns:
            continue
        values = modality_df[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "feature": name,
                "mean_shap": float(np.mean(values)),
                "mean_abs_shap": float(np.mean(np.abs(values))),
                "std_shap": float(np.std(values)),
                "min_shap": float(np.min(values)),
                "max_shap": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_modality_histograms(modality_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes_flat = axes.flatten()
    plot_names = MODALITY_NAMES + ["graph_total", "fragment_total"]
    for ax, name in zip(axes_flat, plot_names):
        values = modality_df[f"{name}_shap"].to_numpy(dtype=np.float64)
        ax.hist(values, bins=min(30, max(10, int(np.sqrt(len(values)) or 10))), alpha=0.85, color="#4472c4")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("SHAP value")
        ax.set_ylabel("Count")
    for ax in axes_flat[len(plot_names):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_modality_beeswarm(modality_df: pd.DataFrame, out_path: Path) -> None:
    shap_values = modality_df[[f"{name}_shap" for name in MODALITY_NAMES]].to_numpy(dtype=np.float64)
    feature_values = modality_df[[f"{name}_value" for name in MODALITY_NAMES]].to_numpy(dtype=np.float64)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values=shap_values,
        features=feature_values,
        feature_names=MODALITY_NAMES,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_modality_importance_bar(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    ordered = summary_df.copy()
    plt.barh(ordered["feature"][::-1], ordered["mean_abs_shap"][::-1], color="#c0504d")
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def within_fragment_dim_shap(
    model: SynergyModel,
    example: ExampleRecord,
    fragment_smiles: Sequence[str],
    which_drug: str,
    device: str,
    nsamples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if which_drug == "a":
        fragment_matrix = np.asarray(example.frag_a, dtype=np.float32)

        def predict_from_flat(flat_batch: np.ndarray) -> np.ndarray:
            preds = []
            for flat in flat_batch:
                updated = ExampleRecord(
                    cell=example.cell,
                    mol_a=example.mol_a,
                    mol_b=example.mol_b,
                    frag_a=flat.reshape(fragment_matrix.shape).astype(np.float32),
                    frag_b=example.frag_b,
                    target=example.target,
                    meta_row=example.meta_row,
                )
                preds.append(
                    predict_single_with_gates(
                        model,
                        updated,
                        modality_gates=np.ones(5, dtype=np.float32),
                        device=device,
                    )
                )
            return np.asarray(preds, dtype=np.float32)

    elif which_drug == "b":
        fragment_matrix = np.asarray(example.frag_b, dtype=np.float32)

        def predict_from_flat(flat_batch: np.ndarray) -> np.ndarray:
            preds = []
            for flat in flat_batch:
                updated = ExampleRecord(
                    cell=example.cell,
                    mol_a=example.mol_a,
                    mol_b=example.mol_b,
                    frag_a=example.frag_a,
                    frag_b=flat.reshape(fragment_matrix.shape).astype(np.float32),
                    target=example.target,
                    meta_row=example.meta_row,
                )
                preds.append(
                    predict_single_with_gates(
                        model,
                        updated,
                        modality_gates=np.ones(5, dtype=np.float32),
                        device=device,
                    )
                )
            return np.asarray(preds, dtype=np.float32)
    else:
        raise ValueError(which_drug)

    flat_input = fragment_matrix.reshape(1, -1)
    background = np.zeros((1, flat_input.shape[1]), dtype=np.float32)
    explainer = shap.KernelExplainer(predict_from_flat, background)
    shap_values = explainer.shap_values(flat_input, nsamples=nsamples)
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[0], dtype=np.float32).reshape(fragment_matrix.shape)
    else:
        shap_array = np.asarray(shap_values, dtype=np.float32).reshape(fragment_matrix.shape)

    dim_rows = []
    fragment_rows = []
    for frag_idx in range(fragment_matrix.shape[0]):
        abs_sum = float(np.abs(shap_array[frag_idx]).sum())
        signed_sum = float(shap_array[frag_idx].sum())
        frag_smiles = fragment_smiles[frag_idx] if frag_idx < len(fragment_smiles) else None
        fragment_rows.append(
            {
                "drug_side": which_drug,
                "fragment_index": frag_idx,
                "fragment_smiles": frag_smiles,
                "fragment_embedding_dim_abs_shap_sum": abs_sum,
                "fragment_embedding_dim_signed_shap_sum": signed_sum,
            }
        )
        for dim_idx in range(fragment_matrix.shape[1]):
            dim_rows.append(
                {
                    "drug_side": which_drug,
                    "fragment_index": frag_idx,
                    "fragment_smiles": frag_smiles,
                    "embedding_dim": dim_idx,
                    "embedding_value": float(fragment_matrix[frag_idx, dim_idx]),
                    "embedding_dim_shap": float(shap_array[frag_idx, dim_idx]),
                    "abs_embedding_dim_shap": float(abs(shap_array[frag_idx, dim_idx])),
                }
            )

    dim_df = pd.DataFrame(dim_rows)
    fragment_df = pd.DataFrame(fragment_rows).sort_values(
        "fragment_embedding_dim_abs_shap_sum", ascending=False
    ).reset_index(drop=True)
    return dim_df, fragment_df


def add_fragment_descriptors(
    fragment_df: pd.DataFrame,
    attention_rows: Sequence[Dict[str, object]],
    top_k: int,
) -> pd.DataFrame:
    attention_lookup = {
        int(row["fragment_index"]): row for row in attention_rows
    }
    rows = []
    for row in fragment_df.itertuples(index=False):
        descriptors = fragment_descriptor_dict(row.fragment_smiles) if row.fragment_smiles else None
        base = {
            "drug_side": row.drug_side,
            "fragment_index": int(row.fragment_index),
            "fragment_smiles": row.fragment_smiles,
            "fragment_embedding_dim_abs_shap_sum": float(row.fragment_embedding_dim_abs_shap_sum),
            "fragment_embedding_dim_signed_shap_sum": float(row.fragment_embedding_dim_signed_shap_sum),
        }
        extra = attention_lookup.get(int(row.fragment_index), {})
        base["attention_weight"] = float(extra.get("attention_weight", np.nan))
        base["leave_one_out_delta"] = float(extra.get("leave_one_out_delta", np.nan))
        base["fragment_shap_value"] = float(extra.get("fragment_shap_value", np.nan))
        if descriptors:
            base.update(descriptors)
        rows.append(base)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fragment_direction"] = np.where(
        out["leave_one_out_delta"] < 0,
        "synergy_decreasing",
        np.where(out["leave_one_out_delta"] > 0, "synergy_increasing", "neutral"),
    )
    out["least_importance_score"] = out["leave_one_out_delta"].abs()
    return out.reset_index(drop=True)


def rank_fragment_descriptors(fragment_df: pd.DataFrame, mode: str, top_k: int) -> pd.DataFrame:
    if fragment_df.empty:
        return fragment_df.copy()
    ranked = fragment_df.copy()
    if mode == "most_harmful":
        ranked = ranked[pd.to_numeric(ranked["leave_one_out_delta"], errors="coerce") < 0].copy()
        if ranked.empty:
            ranked["fragment_rank_mode"] = mode
            return ranked
        ranked = ranked.sort_values(
            ["leave_one_out_delta", "fragment_embedding_dim_signed_shap_sum", "fragment_embedding_dim_abs_shap_sum"],
            ascending=[True, True, False],
        )
    elif mode == "least_important":
        ranked = ranked.sort_values(
            ["least_importance_score", "fragment_embedding_dim_abs_shap_sum"],
            ascending=[True, True],
        )
    elif mode == "most_helpful":
        ranked = ranked[pd.to_numeric(ranked["leave_one_out_delta"], errors="coerce") > 0].copy()
        if ranked.empty:
            ranked["fragment_rank_mode"] = mode
            return ranked
        ranked = ranked.sort_values(
            ["leave_one_out_delta", "fragment_embedding_dim_signed_shap_sum", "fragment_embedding_dim_abs_shap_sum"],
            ascending=[False, False, False],
        )
    else:
        raise ValueError(mode)
    ranked["fragment_rank_mode"] = mode
    return ranked.head(top_k).reset_index(drop=True)


def top_unique_pairs(filtered: pd.DataFrame, top_n: int) -> pd.DataFrame:
    pair_df = filtered.copy()
    pair_df["pair_key"] = pair_df["drug_a"].astype(str) + " || " + pair_df["drug_b"].astype(str)
    pair_df = pair_df.drop_duplicates("pair_key", keep="first").drop(columns="pair_key")
    return pair_df.head(top_n).reset_index(drop=True)


def pair_output_dir(base_dir: Path, row: pd.Series) -> Path:
    safe_drug_a = str(row["drug_a"]).replace("/", "_")
    safe_drug_b = str(row["drug_b"]).replace("/", "_")
    return base_dir / f"{safe_drug_a}__{safe_drug_b}"


def save_pair_fragment_outputs(
    outdir: Path,
    fragment_rows_a: List[Dict[str, object]],
    fragment_rows_b: List[Dict[str, object]],
    fragment_dim_df_a: pd.DataFrame,
    fragment_dim_df_b: pd.DataFrame,
    fragment_dim_summary_a: pd.DataFrame,
    fragment_dim_summary_b: pd.DataFrame,
    top_descriptor_df_a: Dict[str, pd.DataFrame],
    top_descriptor_df_b: Dict[str, pd.DataFrame],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fragment_rows_a).to_csv(outdir / "drug_a_fragments.csv", index=False)
    pd.DataFrame(fragment_rows_b).to_csv(outdir / "drug_b_fragments.csv", index=False)
    fragment_dim_df_a.to_csv(outdir / "drug_a_fragment_embedding_dim_shap.csv", index=False)
    fragment_dim_df_b.to_csv(outdir / "drug_b_fragment_embedding_dim_shap.csv", index=False)
    fragment_dim_summary_a.to_csv(outdir / "drug_a_fragment_importance_from_dims.csv", index=False)
    fragment_dim_summary_b.to_csv(outdir / "drug_b_fragment_importance_from_dims.csv", index=False)
    for mode, df in top_descriptor_df_a.items():
        df.to_csv(outdir / f"drug_a_top_fragment_descriptors_{mode}.csv", index=False)
    for mode, df in top_descriptor_df_b.items():
        df.to_csv(outdir / f"drug_b_top_fragment_descriptors_{mode}.csv", index=False)
    top_descriptor_df_a.get("most_harmful", pd.DataFrame()).to_csv(outdir / "drug_a_top_fragment_descriptors.csv", index=False)
    top_descriptor_df_b.get("most_harmful", pd.DataFrame()).to_csv(outdir / "drug_b_top_fragment_descriptors.csv", index=False)


def analyze_pair_fragments(
    model: SynergyModel,
    example: ExampleRecord,
    fragment_map: Dict[str, List[str]],
    fragment_embeddings: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, object]:
    fragment_smiles_a = [
        frag_smiles
        for frag_smiles in fragment_map.get(str(example.meta_row["drug_a_smiles"]), [])
        if frag_smiles in fragment_embeddings
    ]
    fragment_smiles_b = [
        frag_smiles
        for frag_smiles in fragment_map.get(str(example.meta_row["drug_b_smiles"]), [])
        if frag_smiles in fragment_embeddings
    ]

    selected_prediction = float(example.meta_row["predicted_synergy"])
    attn_a = fragment_attention_weights(model.frag_pool, example.frag_a, args.device)
    attn_b = fragment_attention_weights(model.frag_pool, example.frag_b, args.device)
    loo_a = leave_one_out_fragment_effects(model, example, selected_prediction, args.device, "a")
    loo_b = leave_one_out_fragment_effects(model, example, selected_prediction, args.device, "b")

    frag_shap_a = None
    if example.frag_a.shape[0] <= args.max_exact_fragments:
        frag_shap_a = fragment_shap(model, example, args.device, "a")
    frag_shap_b = None
    if example.frag_b.shape[0] <= args.max_exact_fragments:
        frag_shap_b = fragment_shap(model, example, args.device, "b")

    fragment_rows_a = build_fragment_rows(
        smiles=str(example.meta_row["drug_a_smiles"]),
        fragment_map=fragment_map,
        fragment_embeddings=fragment_embeddings,
        attention=attn_a,
        loo_effects=loo_a,
        shap_payload=frag_shap_a,
    )
    fragment_rows_b = build_fragment_rows(
        smiles=str(example.meta_row["drug_b_smiles"]),
        fragment_map=fragment_map,
        fragment_embeddings=fragment_embeddings,
        attention=attn_b,
        loo_effects=loo_b,
        shap_payload=frag_shap_b,
    )
    fragment_dim_df_a, fragment_dim_summary_a = within_fragment_dim_shap(
        model=model,
        example=example,
        fragment_smiles=fragment_smiles_a,
        which_drug="a",
        device=args.device,
        nsamples=args.fragment_dim_shap_samples,
    )
    fragment_dim_df_b, fragment_dim_summary_b = within_fragment_dim_shap(
        model=model,
        example=example,
        fragment_smiles=fragment_smiles_b,
        which_drug="b",
        device=args.device,
        nsamples=args.fragment_dim_shap_samples,
    )
    descriptor_df_a = add_fragment_descriptors(
        fragment_df=fragment_dim_summary_a,
        attention_rows=fragment_rows_a,
        top_k=args.top_fragment_descriptors,
    )
    descriptor_df_b = add_fragment_descriptors(
        fragment_df=fragment_dim_summary_b,
        attention_rows=fragment_rows_b,
        top_k=args.top_fragment_descriptors,
    )
    top_descriptor_df_a = {
        mode: rank_fragment_descriptors(descriptor_df_a, mode, args.top_fragment_descriptors)
        for mode in ["most_harmful", "least_important", "most_helpful"]
    }
    top_descriptor_df_b = {
        mode: rank_fragment_descriptors(descriptor_df_b, mode, args.top_fragment_descriptors)
        for mode in ["most_harmful", "least_important", "most_helpful"]
    }
    return {
        "fragment_rows_a": fragment_rows_a,
        "fragment_rows_b": fragment_rows_b,
        "fragment_dim_df_a": fragment_dim_df_a,
        "fragment_dim_df_b": fragment_dim_df_b,
        "fragment_dim_summary_a": fragment_dim_summary_a,
        "fragment_dim_summary_b": fragment_dim_summary_b,
        "top_descriptor_df_a": top_descriptor_df_a,
        "top_descriptor_df_b": top_descriptor_df_b,
    }


def _render_mol_image(smiles: Optional[str], highlight_smiles: Optional[str] = None, size: Tuple[int, int] = (320, 320)) -> np.ndarray:
    if not smiles:
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    highlight_atoms: List[int] = []
    if highlight_smiles:
        frag = Chem.MolFromSmiles(str(highlight_smiles))
        if frag is not None:
            matches = mol.GetSubstructMatches(frag)
            if matches:
                highlight_atoms = list(matches[0])
    image = Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms)
    return np.asarray(image)


def _save_top_harmful_fragment_report(top_harmful_df: pd.DataFrame, out_path: Path) -> None:
    if top_harmful_df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for row in top_harmful_df.itertuples(index=False):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            panels = [
                (_render_mol_image(getattr(row, "drug_a_smiles", None)), f"Drug A: {row.drug_a}"),
                (_render_mol_image(getattr(row, "drug_b_smiles", None)), f"Drug B: {row.drug_b}"),
                (
                    _render_mol_image(getattr(row, "source_molecule_smiles", None), getattr(row, "fragment_smiles", None)),
                    f"Source molecule ({row.source_molecule_name}) with harmful fragment highlighted",
                ),
                (_render_mol_image(getattr(row, "fragment_smiles", None)), f"Harmful fragment: {row.fragment_smiles}"),
            ]
            for ax, (img, title) in zip(axes.flatten(), panels):
                ax.imshow(img)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            fig.suptitle(
                f"pair_rank={row.pair_rank} | {row.drug_a} + {row.drug_b} | side={row.drug_side} | "
                f"leave_one_out_delta={float(row.leave_one_out_delta):.4f} | predicted_synergy={float(row.predicted_synergy):.4f}",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, dpi=220)
            plt.close(fig)


def save_outputs(
    outdir: Path,
    cell_line_name: str,
    modality_df: pd.DataFrame,
    modality_summary_df: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    top_pairs_df: pd.DataFrame,
    top_fragments_all_df: pd.DataFrame,
    pair_summaries: List[Dict[str, object]],
    harmful_fragment_top_n: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    candidate_rows.to_csv(outdir / "candidate_rows.csv", index=False)
    modality_df.to_csv(outdir / "modality_shap_all_pairs.csv", index=False)
    modality_summary_df.to_csv(outdir / "modality_shap_summary.csv", index=False)
    top_pairs_df.to_csv(outdir / "top_pairs_for_fragment_analysis.csv", index=False)
    top_fragments_all_df.to_csv(outdir / "top_fragments_across_top_pairs.csv", index=False)
    top_harmful_df = pd.DataFrame()
    if not top_fragments_all_df.empty and "fragment_rank_mode" in top_fragments_all_df.columns:
        for mode, mode_df in top_fragments_all_df.groupby("fragment_rank_mode", dropna=False):
            mode_df.to_csv(outdir / f"top_fragments_across_top_pairs_{mode}.csv", index=False)
        top_harmful_df = top_fragments_all_df[
            (top_fragments_all_df["fragment_rank_mode"] == "most_harmful")
            & (pd.to_numeric(top_fragments_all_df["leave_one_out_delta"], errors="coerce") < 0)
        ].sort_values("leave_one_out_delta", ascending=True).head(harmful_fragment_top_n).reset_index(drop=True)
        top_harmful_df.to_csv(outdir / f"top_{harmful_fragment_top_n}_harmful_fragments.csv", index=False)
        _save_top_harmful_fragment_report(top_harmful_df, outdir / f"top_{harmful_fragment_top_n}_harmful_fragments_report.pdf")

    plot_modality_beeswarm(modality_df, outdir / "modality_shap_beeswarm.png")
    plot_modality_histograms(modality_df, outdir / "modality_shap_histograms.png")
    plot_modality_importance_bar(modality_summary_df, outdir / "modality_shap_mean_abs_bar.png")

    payload = {
        "cell_line_name": cell_line_name,
        "n_pairs_analyzed": int(len(modality_df)),
        "n_top_pairs_with_fragment_analysis": int(len(top_pairs_df)),
        "cell_line_modality_summary": modality_summary_df.to_dict(orient="records"),
        "top_pairs_for_fragment_analysis": top_pairs_df.to_dict(orient="records"),
        "top_fragments_across_top_pairs": top_fragments_all_df.to_dict(orient="records"),
        "pair_fragment_summaries": pair_summaries,
    }
    (outdir / "explanation.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint = load_checkpoint(Path(args.checkpoint), args.device)
    model = build_model(checkpoint, args.device)
    examples = build_examples_bundle(args, checkpoint)
    fragment_map = examples["fragment_map"]
    fragment_embeddings = examples["fragment_embeddings"]

    filtered = filter_examples(examples, args, model)
    modality_df = compute_cell_line_modality_shap(model, examples, filtered, args.device)
    modality_summary_df = summarize_cell_line_modality_shap(modality_df)
    top_pairs_df = top_unique_pairs(filtered, args.fragment_detail_top_pairs)
    pair_details_root = Path(args.outdir) / str(filtered.iloc[0]["cell_line_name"]).replace("/", "_") / "pair_details"
    top_fragment_rows_all = []
    pair_summaries: List[Dict[str, object]] = []
    for pair_idx in range(len(top_pairs_df)):
        pair_row = top_pairs_df.iloc[pair_idx]
        pair_example = select_example(
            examples,
            top_pairs_df,
            argparse.Namespace(select_by="index", example_index=pair_idx),
        )
        pair_deep = analyze_pair_fragments(
            model=model,
            example=pair_example,
            fragment_map=fragment_map,
            fragment_embeddings=fragment_embeddings,
            args=args,
        )
        pair_dir = pair_output_dir(pair_details_root, pair_row)
        save_pair_fragment_outputs(
            outdir=pair_dir,
            fragment_rows_a=pair_deep["fragment_rows_a"],
            fragment_rows_b=pair_deep["fragment_rows_b"],
            fragment_dim_df_a=pair_deep["fragment_dim_df_a"],
            fragment_dim_df_b=pair_deep["fragment_dim_df_b"],
            fragment_dim_summary_a=pair_deep["fragment_dim_summary_a"],
            fragment_dim_summary_b=pair_deep["fragment_dim_summary_b"],
            top_descriptor_df_a=pair_deep["top_descriptor_df_a"],
            top_descriptor_df_b=pair_deep["top_descriptor_df_b"],
        )
        for side, ranked_map in [("a", pair_deep["top_descriptor_df_a"]), ("b", pair_deep["top_descriptor_df_b"] )]:
            for mode, df in ranked_map.items():
                if df.empty:
                    continue
                enriched = df.copy()
                enriched["pair_rank"] = pair_idx + 1
                enriched["drug_a"] = pair_row["drug_a"]
                enriched["drug_b"] = pair_row["drug_b"]
                enriched["predicted_synergy"] = float(pair_row["predicted_synergy"])
                enriched["observed_synergy"] = float(pair_row["y_loewe"])
                enriched["detail_dir"] = str(pair_dir)
                enriched["drug_side"] = side
                enriched["fragment_rank_mode"] = mode
                enriched["drug_a_smiles"] = pair_row["drug_a_smiles"]
                enriched["drug_b_smiles"] = pair_row["drug_b_smiles"]
                if side == "a":
                    enriched["source_molecule_name"] = pair_row["drug_a"]
                    enriched["source_molecule_smiles"] = pair_row["drug_a_smiles"]
                    enriched["other_molecule_name"] = pair_row["drug_b"]
                    enriched["other_molecule_smiles"] = pair_row["drug_b_smiles"]
                else:
                    enriched["source_molecule_name"] = pair_row["drug_b"]
                    enriched["source_molecule_smiles"] = pair_row["drug_b_smiles"]
                    enriched["other_molecule_name"] = pair_row["drug_a"]
                    enriched["other_molecule_smiles"] = pair_row["drug_a_smiles"]
                top_fragment_rows_all.append(enriched)
        pair_summaries.append(
            {
                "pair_rank": pair_idx + 1,
                "drug_a": pair_row["drug_a"],
                "drug_b": pair_row["drug_b"],
                "predicted_synergy": float(pair_row["predicted_synergy"]),
                "observed_synergy": float(pair_row["y_loewe"]),
                "detail_dir": str(pair_dir),
                "drug_a_top_fragments": {
                    mode: df.to_dict(orient="records") for mode, df in pair_deep["top_descriptor_df_a"].items()
                },
                "drug_b_top_fragments": {
                    mode: df.to_dict(orient="records") for mode, df in pair_deep["top_descriptor_df_b"].items()
                },
            }
        )

    top_fragments_all_df = (
        pd.concat(top_fragment_rows_all, axis=0, ignore_index=True)
        if top_fragment_rows_all
        else pd.DataFrame()
    )
    if not top_fragments_all_df.empty:
        top_fragments_all_df = top_fragments_all_df.reset_index(drop=True)

    safe_cell = str(filtered.iloc[0]["cell_line_name"]).replace("/", "_")
    outdir = Path(args.outdir) / safe_cell
    save_outputs(
        outdir=outdir,
        cell_line_name=str(filtered.iloc[0]["cell_line_name"]),
        modality_df=modality_df,
        modality_summary_df=modality_summary_df,
        candidate_rows=filtered,
        top_pairs_df=top_pairs_df,
        top_fragments_all_df=top_fragments_all_df,
        pair_summaries=pair_summaries,
        harmful_fragment_top_n=args.harmful_fragment_top_n,
    )

    print(f"Selected cell line: {filtered.iloc[0]['cell_line_name']}")
    print(f"Pairs analyzed for this cell line: {len(filtered)}")
    print(f"Top unique pairs with deep fragment analysis: {len(top_pairs_df)}")
    print("Cell-line-wide modality importance by mean |SHAP|:")
    for row in modality_summary_df.head(7).itertuples(index=False):
        print(f"  {row.feature}: {float(row.mean_abs_shap):.6f}")
    if not top_fragments_all_df.empty:
        harmful_df = top_fragments_all_df[
            (top_fragments_all_df["fragment_rank_mode"] == "most_harmful")
            & (pd.to_numeric(top_fragments_all_df["leave_one_out_delta"], errors="coerce") < 0)
        ].sort_values("leave_one_out_delta", ascending=True)
        print("Top harmful fragments across top-ranked pairs:")
        for row in harmful_df.head(10).itertuples(index=False):
            print(
                f"  rank_pair={row.pair_rank} {row.drug_a}+{row.drug_b} "
                f"side={row.drug_side} fragment={row.fragment_index} "
                f"loo={float(row.leave_one_out_delta):.6f} "
                f"smiles={row.fragment_smiles}"
            )
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
