#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from synergy_shap_analysis import (
    ExampleRecord,
    build_examples_bundle,
    build_model,
    filter_examples,
    load_checkpoint,
    predict_single_with_gates,
    select_example,
)
from synergy_prediction import set_seed


DESCRIPTOR_NAMES = [
    "heavy_atoms",
    "mol_wt",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "rings",
    "aromatic_rings",
    "rot_bonds",
    "fraction_csp3",
]

SIMILARITY_NAMES = [
    "morgan_tanimoto",
    "morgan_dice",
    "rdk_tanimoto",
    "atom_pair_tanimoto",
]

BASE_FEATURE_NAMES = [
    "source_fragment_importance",
    "source_attention_weight",
    "source_leave_one_out_delta",
    *[f"source_{name}" for name in DESCRIPTOR_NAMES],
    *[f"candidate_{name}" for name in DESCRIPTOR_NAMES],
    *[f"delta_{name}" for name in DESCRIPTOR_NAMES],
    *SIMILARITY_NAMES,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a descriptor-based symbolic-regression surrogate for fragment replacement "
            "effects on synergy, using the trained synergy model as a teacher oracle."
        )
    )
    parser.add_argument("--checkpoint", default="oneil_synergy_runs/model.pt")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--molecule-embeddings", default="molecule_smiles_embeddings.pkl")
    parser.add_argument("--fragment-embeddings", default="fragment_smiles_embeddings.pkl")
    parser.add_argument("--fragment-map", default="smiles_to_fragments.pkl")
    parser.add_argument("--drug-smiles-map", default="everything_drug_smiles_map.csv")
    parser.add_argument("--cell-line", required=True)
    parser.add_argument(
        "--top-fragments-table",
        default=None,
        help="Defaults to oneil_synergy_shap/<cell-line>/top_fragments_across_top_pairs.csv",
    )
    parser.add_argument(
        "--replace-drug",
        choices=["a", "b", "both"],
        default="both",
        help="Which side of the pair to generate replacement candidates for from the top-fragments table.",
    )
    parser.add_argument(
        "--fragment-rank-column",
        choices=["fragment_shap_value", "leave_one_out_delta", "attention_weight"],
        default="fragment_shap_value",
    )
    parser.add_argument(
        "--seed-selection-mode",
        choices=["most_harmful", "least_important", "most_helpful", "table_order"],
        default="most_harmful",
        help=(
            "How to choose fragment seeds from the top-fragments table. "
            "`most_harmful` targets fragments pulling synergy down, `least_important` targets weak contributors."
        ),
    )
    parser.add_argument(
        "--harmful-seed-threshold",
        type=float,
        default=0.0,
        help=(
            "When --seed-selection-mode=most_harmful, keep only seeds with leave_one_out_delta below this value. "
            "Default 0.0 means strictly negative leave-one-out seeds only."
        ),
    )
    parser.add_argument(
        "--allow-nonnegative-harmful-seeds",
        action="store_true",
        help=(
            "If set, do not enforce negative leave-one-out filtering for most_harmful seed selection. "
            "By default the symbolic stage now keeps only strictly harmful seeds."
        ),
    )
    parser.add_argument("--top-k-fragments", type=int, default=1)
    parser.add_argument("--max-candidates-per-fragment", type=int, default=0, help="Maximum replacement candidates to keep per fragment after chemistry filtering. Use 0 for no cap.")
    parser.add_argument("--min-morgan-similarity", type=float, default=0.30)
    parser.add_argument("--max-heavy-atom-delta", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=3000)
    parser.add_argument("--generations", type=int, default=35)
    parser.add_argument("--parsimony-coefficient", type=float, default=0.005)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-fragment-seeds", type=int, default=0, help="Maximum number of fragment seeds to keep after ranking. Use 0 for no cap.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--draw-top-candidates", type=int, default=12)
    parser.add_argument("--outdir", default="bioisostere_symbolic_runs")

    # New controls for stronger signal extraction and more stable symbolic search.
    parser.add_argument(
        "--delta-filter-mode",
        choices=["none", "absolute", "std"],
        default="none",
        help="How to filter low-signal teacher delta examples before fitting.",
    )
    parser.add_argument(
        "--delta-filter-threshold",
        type=float,
        default=0.5,
        help="Threshold used when --delta-filter-mode=absolute.",
    )
    parser.add_argument(
        "--feature-top-k",
        type=int,
        default=15,
        help="Number of train-selected features to keep for symbolic regression.",
    )
    parser.add_argument(
        "--clip-scaled-features",
        type=float,
        default=8.0,
        help="Clip scaled features to [-value, value] after robust scaling.",
    )
    parser.add_argument(
        "--sample-weight-power",
        type=float,
        default=0.0,
        help="If >0, sample weights are proportional to |teacher_delta_synergy|^power. Set 0 to disable.",
    )

    parser.add_argument(
        "--target-clip",
        type=float,
        default=10.0,
        help="Clip training targets to [-value, value] before fitting; set <=0 to disable.",
    )
    parser.add_argument(
        "--drop-source-features",
        action="store_true",
        help=(
            "Exclude all source-fragment and delta descriptors from symbolic regression features, "
            "keeping only candidate descriptors, raw similarity scores, and teacher_original_abs. "
            "Forces gplearn to learn candidate-discriminating rules rather than collapsing to "
            "source-fragment importance as a proxy."
        ),
    )
    return parser.parse_args()


# --- chemistry helpers -------------------------------------------------------------------------

def fragment_descriptor_dict(smiles: str) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "rot_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


def fingerprint_bundle(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "morgan": AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048),
        "rdk": Chem.RDKFingerprint(mol),
        "atom_pair": rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048),
    }


def similarity_features(source_smiles: str, candidate_smiles: str) -> Optional[Dict[str, float]]:
    src = fingerprint_bundle(source_smiles)
    cand = fingerprint_bundle(candidate_smiles)
    if src is None or cand is None:
        return None
    return {
        "morgan_tanimoto": float(DataStructs.TanimotoSimilarity(src["morgan"], cand["morgan"])),
        "morgan_dice": float(DataStructs.DiceSimilarity(src["morgan"], cand["morgan"])),
        "rdk_tanimoto": float(DataStructs.TanimotoSimilarity(src["rdk"], cand["rdk"])),
        "atom_pair_tanimoto": float(DataStructs.TanimotoSimilarity(src["atom_pair"], cand["atom_pair"])),
    }


def replace_fragment_matrix(fragment_matrix: np.ndarray, fragment_index: int, replacement_embedding: np.ndarray) -> np.ndarray:
    updated = np.asarray(fragment_matrix, dtype=np.float32).copy()
    updated[fragment_index] = np.asarray(replacement_embedding, dtype=np.float32)
    return updated


def build_replaced_example(
    example: ExampleRecord,
    replace_drug: str,
    fragment_index: int,
    replacement_embedding: np.ndarray,
) -> ExampleRecord:
    if replace_drug == "a":
        return ExampleRecord(
            cell=example.cell,
            mol_a=example.mol_a,
            mol_b=example.mol_b,
            frag_a=replace_fragment_matrix(example.frag_a, fragment_index, replacement_embedding),
            frag_b=example.frag_b,
            target=example.target,
            meta_row=example.meta_row,
        )
    if replace_drug == "b":
        return ExampleRecord(
            cell=example.cell,
            mol_a=example.mol_a,
            mol_b=example.mol_b,
            frag_a=example.frag_a,
            frag_b=replace_fragment_matrix(example.frag_b, fragment_index, replacement_embedding),
            target=example.target,
            meta_row=example.meta_row,
        )
    raise ValueError(replace_drug)


# --- dataset building --------------------------------------------------------------------------

def build_fragment_pool(
    fragment_embeddings: Dict[str, np.ndarray],
    min_similarity: float,
    max_heavy_atom_delta: int,
    source_fragment_smiles: str,
    max_candidates: int,
) -> pd.DataFrame:
    src_desc = fragment_descriptor_dict(source_fragment_smiles)
    if src_desc is None:
        raise ValueError(f"Could not parse source fragment SMILES: {source_fragment_smiles}")

    rows = []
    for cand_smiles in fragment_embeddings.keys():
        if cand_smiles == source_fragment_smiles:
            continue
        cand_desc = fragment_descriptor_dict(cand_smiles)
        sim = similarity_features(source_fragment_smiles, cand_smiles)
        if cand_desc is None or sim is None:
            continue
        if sim["morgan_tanimoto"] < min_similarity:
            continue
        if abs(cand_desc["heavy_atoms"] - src_desc["heavy_atoms"]) > max_heavy_atom_delta:
            continue
        row = {
            "candidate_fragment_smiles": cand_smiles,
            **{f"source_{k}": float(v) for k, v in src_desc.items()},
            **{f"candidate_{k}": float(v) for k, v in cand_desc.items()},
            **{f"delta_{k}": float(cand_desc[k] - src_desc[k]) for k in DESCRIPTOR_NAMES},
            **sim,
        }
        rows.append(row)

    pool_df = pd.DataFrame(rows)
    if pool_df.empty:
        return pool_df
    ranked_pool = pool_df.sort_values(["morgan_tanimoto", "atom_pair_tanimoto"], ascending=False).reset_index(drop=True)
    if max_candidates is not None and int(max_candidates) > 0:
        ranked_pool = ranked_pool.head(int(max_candidates)).reset_index(drop=True)
    return ranked_pool


def build_training_table(
    model,
    example: ExampleRecord,
    important_fragments: pd.DataFrame,
    fragment_embeddings: Dict[str, np.ndarray],
    device: str,
    min_similarity: float,
    max_heavy_atom_delta: int,
    max_candidates_per_fragment: int,
) -> pd.DataFrame:
    base_prediction = predict_single_with_gates(model, example, np.ones(5, dtype=np.float32), device)
    training_rows: List[Dict[str, object]] = []

    for row in important_fragments.itertuples(index=False):
        fragment_pool = build_fragment_pool(
            fragment_embeddings=fragment_embeddings,
            min_similarity=min_similarity,
            max_heavy_atom_delta=max_heavy_atom_delta,
            source_fragment_smiles=row.fragment_smiles,
            max_candidates=max_candidates_per_fragment,
        )
        if fragment_pool.empty:
            continue

        for cand in fragment_pool.itertuples(index=False):
            replaced_example = build_replaced_example(
                example=example,
                replace_drug=row.replace_drug,
                fragment_index=int(row.fragment_index),
                replacement_embedding=fragment_embeddings[cand.candidate_fragment_smiles],
            )
            replaced_prediction = predict_single_with_gates(model, replaced_example, np.ones(5, dtype=np.float32), device)
            training_rows.append(
                {
                    "cell_line_name": example.meta_row["cell_line_name"],
                    "drug_a": example.meta_row["drug_a"],
                    "drug_b": example.meta_row["drug_b"],
                    "drug_a_smiles": example.meta_row["drug_a_smiles"],
                    "drug_b_smiles": example.meta_row["drug_b_smiles"],
                    "replace_drug": row.replace_drug,
                    "fragment_index": int(row.fragment_index),
                    "source_fragment_smiles": row.fragment_smiles,
                    "candidate_fragment_smiles": cand.candidate_fragment_smiles,
                    "source_fragment_importance": float(
                        row.fragment_shap_value if pd.notna(row.fragment_shap_value) else row.leave_one_out_delta
                    ),
                    "source_attention_weight": float(row.attention_weight),
                    "source_leave_one_out_delta": float(row.leave_one_out_delta),
                    "teacher_original_synergy": float(base_prediction),
                    "teacher_replaced_synergy": float(replaced_prediction),
                    "teacher_delta_synergy": float(replaced_prediction - base_prediction),
                    **{name: getattr(cand, name) for name in list(cand._fields) if name not in {"Index"}},
                }
            )

    return pd.DataFrame(training_rows)


def resolve_top_fragments_table(args: argparse.Namespace) -> Path:
    if args.top_fragments_table:
        return Path(args.top_fragments_table)
    return Path("oneil_synergy_shap") / args.cell_line / "top_fragments_across_top_pairs.csv"


def rank_fragment_seeds(
    top_fragments_df: pd.DataFrame,
    mode: str,
    max_fragment_seeds: int,
    harmful_seed_threshold: float = 0.0,
    allow_nonnegative_harmful_seeds: bool = False,
) -> pd.DataFrame:
    ranked = top_fragments_df.copy()
    if ranked.empty:
        return ranked

    if "fragment_rank_mode" in ranked.columns and mode != "table_order":
        preferred = ranked[ranked["fragment_rank_mode"].astype(str) == mode].copy()
        if not preferred.empty:
            ranked = preferred

    if mode == "most_harmful":
        if not allow_nonnegative_harmful_seeds:
            ranked = ranked[pd.to_numeric(ranked["leave_one_out_delta"], errors="coerce") < float(harmful_seed_threshold)].copy()
        ranked = ranked.sort_values(
            ["leave_one_out_delta", "fragment_embedding_dim_signed_shap_sum", "fragment_embedding_dim_abs_shap_sum"],
            ascending=[True, True, False],
        )
    elif mode == "least_important":
        if "least_importance_score" in ranked.columns:
            ranked = ranked.sort_values(["least_importance_score", "fragment_embedding_dim_abs_shap_sum"], ascending=[True, True])
        else:
            ranked = ranked.assign(_least_importance_score=ranked["leave_one_out_delta"].abs()).sort_values(
                ["_least_importance_score", "fragment_embedding_dim_abs_shap_sum"],
                ascending=[True, True],
            )
    elif mode == "most_helpful":
        ranked = ranked.sort_values(
            ["leave_one_out_delta", "fragment_embedding_dim_signed_shap_sum", "fragment_embedding_dim_abs_shap_sum"],
            ascending=[False, False, False],
        )
    elif mode == "table_order":
        ranked = ranked.copy()
    else:
        raise ValueError(mode)

    if "_least_importance_score" in ranked.columns:
        ranked = ranked.drop(columns=["_least_importance_score"])
    if max_fragment_seeds is not None and int(max_fragment_seeds) > 0:
        ranked = ranked.head(int(max_fragment_seeds))
    return ranked.reset_index(drop=True)


def pick_pair_example(examples: Dict[str, object], model, args: argparse.Namespace, drug_a: str, drug_b: str) -> ExampleRecord:
    filter_args = argparse.Namespace(
        cell_line=args.cell_line,
        drug_a=None,
        drug_b=None,
        pair_order_sensitive=False,
        select_by="predicted_top",
        example_index=0,
        device=args.device,
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
            raise ValueError(f"Pair not found in cell line {args.cell_line}: {drug_a} + {drug_b}")
        pair_rows = swapped
    pair_rows = pair_rows.sort_values("predicted_synergy", ascending=False).reset_index(drop=True)
    return select_example(examples, pair_rows, argparse.Namespace(select_by="index", example_index=0))


# --- feature engineering -----------------------------------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in DESCRIPTOR_NAMES:
        delta_col = f"delta_{col}"
        src_col = f"source_{col}"
        cand_col = f"candidate_{col}"

        if delta_col in out.columns:
            out[f"abs_{delta_col}"] = out[delta_col].abs()
            out[f"sq_{delta_col}"] = out[delta_col] ** 2
            out[f"log1p_abs_{delta_col}"] = np.log1p(out[delta_col].abs())
            out[f"signed_sim_{delta_col}"] = out[delta_col] * out["morgan_tanimoto"]
            out[f"importance_x_{delta_col}"] = out[delta_col] * out["source_fragment_importance"]
            out[f"attention_x_{delta_col}"] = out[delta_col] * out["source_attention_weight"]
            out[f"loo_x_{delta_col}"] = out[delta_col] * out["source_leave_one_out_delta"]

        if src_col in out.columns and cand_col in out.columns:
            out[f"ratio_{col}"] = out[cand_col] / (out[src_col].abs() + 1e-6)
            out[f"product_{col}"] = out[cand_col] * out[src_col]

    out["sim_x_importance"] = out["morgan_tanimoto"] * out["source_fragment_importance"]
    out["sim_x_attention"] = out["morgan_tanimoto"] * out["source_attention_weight"]
    out["sim_x_loo"] = out["morgan_tanimoto"] * out["source_leave_one_out_delta"]
    out["teacher_original_abs"] = out["teacher_original_synergy"].abs()
    out["teacher_replaced_abs"] = out["teacher_replaced_synergy"].abs()

    return out.replace([np.inf, -np.inf], np.nan)


def candidate_feature_columns(training_df: pd.DataFrame, drop_source_features: bool = False) -> List[str]:
    excluded = {
        "cell_line_name",
        "drug_a",
        "drug_b",
        "drug_a_smiles",
        "drug_b_smiles",
        "replace_drug",
        "fragment_index",
        "source_fragment_smiles",
        "candidate_fragment_smiles",
        "teacher_delta_synergy",
        "teacher_original_synergy",
        "teacher_replaced_synergy",
        "pair_rank",
        "seed_fragment_abs_dim_shap",
        "symbolic_pred_delta_synergy",
        "symbolic_error",
        "split",
    }

    # Prefixes/patterns that encode source-fragment properties or source-candidate deltas.
    # Kept when drop_source_features=False; stripped when True.
    _SOURCE_PREFIXES = (
        "source_",       # direct source descriptors and importance scores
        "delta_",        # candidate - source  (encodes source)
        "abs_delta_",    # |delta|
        "sq_delta_",     # delta²
        "log1p_abs_delta_",
        "signed_sim_",   # delta * similarity
        "loo_x_",        # source_loo * delta  interaction
        "importance_x_", # source_importance * delta  interaction
        "attention_x_",  # source_attention * delta  interaction
        "sim_x_loo",     # similarity * source_loo
        "sim_x_importance",
        "sim_x_attention",
        "ratio_",        # candidate / source
        "product_",      # candidate * source
        "teacher_replaced_abs",  # depends on source context implicitly
    )

    cols = []
    for col in training_df.columns:
        if col in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(training_df[col]):
            continue
        if drop_source_features and any(col == p or col.startswith(p) for p in _SOURCE_PREFIXES):
            continue
        cols.append(col)
    return cols


def apply_delta_filter(df: pd.DataFrame, mode: str, threshold: float) -> pd.DataFrame:
    if mode == "none":
        return df.copy()
    y = df["teacher_delta_synergy"].astype(float)
    if mode == "absolute":
        keep = y.abs() > float(threshold)
    elif mode == "std":
        std = float(y.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            return df.copy()
        keep = y.abs() > std
    else:
        raise ValueError(mode)
    filtered = df.loc[keep].copy()
    return filtered if not filtered.empty else df.copy()


def rank_features_by_train_signal(X_train_df: pd.DataFrame, y_train: np.ndarray, top_k: int) -> List[str]:
    scores = {}
    target = pd.Series(y_train, index=X_train_df.index)
    for col in X_train_df.columns:
        series = pd.to_numeric(X_train_df[col], errors="coerce")
        if series.nunique(dropna=True) <= 1:
            continue
        corr = series.corr(target)
        if pd.notna(corr):
            scores[col] = abs(float(corr))
    if not scores:
        return list(X_train_df.columns)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[: max(1, min(top_k, len(ranked)))]]


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pcc": pcc}


def replace_program_tokens(program_str: str, feature_names: Sequence[str]) -> str:
    named = str(program_str)
    for idx in reversed(range(len(feature_names))):
        named = named.replace(f"X{idx}", f"z_{feature_names[idx]}")
    return named


def scaler_summary_df(feature_names: Sequence[str], scaler: RobustScaler) -> pd.DataFrame:
    center = getattr(scaler, "center_", np.zeros(len(feature_names), dtype=float))
    scale = getattr(scaler, "scale_", np.ones(len(feature_names), dtype=float))
    safe_scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return pd.DataFrame({"feature": list(feature_names), "center": center.astype(float), "scale": safe_scale.astype(float)})


def build_final_equation(named_program: str, scaler_df: pd.DataFrame) -> str:
    lines = ["delta_synergy = " + named_program, "", "where"]
    for row in scaler_df.itertuples(index=False):
        lines.append(f"z_{row.feature} = ({row.feature} - {row.center:.10g}) / {row.scale:.10g}")
    return "\n".join(lines) + "\n"


def fit_symbolic_regression(training_df: pd.DataFrame, args: argparse.Namespace):
    # Engineer features once, then do every potentially leaky operation using TRAIN ONLY.
    engineered_df = add_engineered_features(training_df).replace([np.inf, -np.inf], np.nan).copy()

    x_cols_all = candidate_feature_columns(engineered_df, drop_source_features=getattr(args, "drop_source_features", False))
    if not x_cols_all:
        raise ValueError("No numeric feature columns available for symbolic regression.")

    y_all = engineered_df["teacher_delta_synergy"].astype(float).to_numpy()
    indices = np.arange(len(engineered_df))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_fraction, random_state=args.random_state)
    val_fraction_adjusted = args.val_fraction / max(1.0 - args.test_fraction, 1e-6)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_fraction_adjusted, random_state=args.random_state)

    # Impute using train-only medians to avoid leakage.
    train_medians = engineered_df.iloc[train_idx][x_cols_all].median(numeric_only=True)
    X_df = engineered_df[x_cols_all].copy().fillna(train_medians).fillna(0.0)

    # Optional low-signal filtering is TRAIN-ONLY. Validation and test stay untouched.
    y_train_raw_full = y_all[train_idx]
    if args.delta_filter_mode == "none":
        keep_train = np.ones(len(train_idx), dtype=bool)
    elif args.delta_filter_mode == "absolute":
        keep_train = np.abs(y_train_raw_full) > float(args.delta_filter_threshold)
    elif args.delta_filter_mode == "std":
        train_std = float(np.std(y_train_raw_full))
        keep_train = np.abs(y_train_raw_full) > train_std if np.isfinite(train_std) and train_std > 0 else np.ones(len(train_idx), dtype=bool)
    else:
        raise ValueError(args.delta_filter_mode)
    if not np.any(keep_train):
        keep_train = np.ones(len(train_idx), dtype=bool)

    train_fit_idx = train_idx[keep_train]

    # Feature ranking uses TRAIN-FIT rows only.
    X_train_rank_df = X_df.iloc[train_fit_idx].copy()
    y_train_rank = y_all[train_fit_idx]
    selected_cols = rank_features_by_train_signal(X_train_rank_df, y_train_rank, args.feature_top_k)

    X_train_df = X_df.iloc[train_fit_idx][selected_cols].copy()
    X_val_df = X_df.iloc[val_idx][selected_cols].copy()
    X_test_df = X_df.iloc[test_idx][selected_cols].copy()
    X_all_df = X_df[selected_cols].copy()

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_val = scaler.transform(X_val_df)
    X_test = scaler.transform(X_test_df)
    X_all = scaler.transform(X_all_df)

    if args.clip_scaled_features is not None and args.clip_scaled_features > 0:
        clip_v = float(args.clip_scaled_features)
        X_train = np.clip(X_train, -clip_v, clip_v)
        X_val = np.clip(X_val, -clip_v, clip_v)
        X_test = np.clip(X_test, -clip_v, clip_v)
        X_all = np.clip(X_all, -clip_v, clip_v)

    # Keep raw targets for evaluation; use clipped+normalized train targets for fitting stability.
    # If target clipping is active, apply the same clip to val/test so evaluation is consistent
    # with what the model was trained to predict (avoids penalising it for not predicting extremes
    # it never saw during training).
    y_train_raw = y_all[train_fit_idx].astype(np.float32)
    y_val_raw = y_all[val_idx].astype(np.float32)
    y_test_raw = y_all[test_idx].astype(np.float32)

    y_train_fit = y_train_raw.copy()
    if getattr(args, "target_clip", 0.0) and float(args.target_clip) > 0:
        clip_t = float(args.target_clip)
        y_train_fit = np.clip(y_train_fit, -clip_t, clip_t)
        y_val_raw = np.clip(y_val_raw, -clip_t, clip_t)
        y_test_raw = np.clip(y_test_raw, -clip_t, clip_t)

    y_mean = float(np.mean(y_train_fit))
    y_std = float(np.std(y_train_fit))
    if (not np.isfinite(y_std)) or y_std < 1e-6:
        y_std = 1.0
    y_train = ((y_train_fit - y_mean) / y_std).astype(np.float32)

    fit_kwargs = {}
    if float(args.sample_weight_power) > 0:
        fit_kwargs["sample_weight"] = np.power(np.abs(y_train_raw) + 1e-3, float(args.sample_weight_power))

    reg = SymbolicRegressor(
        population_size=args.population_size,
        generations=args.generations,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min"),
        parsimony_coefficient=args.parsimony_coefficient,
        stopping_criteria=0.001,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        random_state=args.random_state,
    )
    reg.fit(X_train, y_train, **fit_kwargs)

    pred_train_scaled = reg.predict(X_train)
    pred_val_scaled = reg.predict(X_val)
    pred_test_scaled = reg.predict(X_test)
    pred_all_scaled = reg.predict(X_all)

    pred_train = pred_train_scaled * y_std + y_mean
    pred_val = pred_val_scaled * y_std + y_mean
    pred_test = pred_test_scaled * y_std + y_mean
    pred_all = pred_all_scaled * y_std + y_mean

    split_labels = np.empty(len(engineered_df), dtype=object)
    split_labels[:] = "unused"
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"
    split_labels[train_fit_idx] = "train_fit"

    raw_program = str(reg._program)
    named_program = replace_program_tokens(raw_program, selected_cols)
    scaler_df = scaler_summary_df(selected_cols, scaler)
    final_equation = build_final_equation(named_program, scaler_df)

    metrics = {
        "train": metric_dict(y_train_raw, pred_train),
        "val": metric_dict(y_val_raw, pred_val),
        "test": metric_dict(y_test_raw, pred_test),
    }

    engineered_df = engineered_df.copy()
    engineered_df["symbolic_pred_delta_synergy"] = pred_all
    engineered_df["symbolic_error"] = engineered_df["symbolic_pred_delta_synergy"] - engineered_df["teacher_delta_synergy"]
    engineered_df["split"] = split_labels

    return {
        "regressor": reg,
        "feature_columns": selected_cols,
        "pred_all": pred_all,
        "split_labels": split_labels,
        "metrics": metrics,
        "raw_equation": raw_program,
        "named_equation": named_program,
        "final_equation": final_equation,
        "scaler_df": scaler_df,
        "training_df_filtered": engineered_df,
        "n_rows_before_filter": int(len(training_df)),
        "n_rows_after_filter": int(len(engineered_df)),
        "n_train_rows_full": int(len(train_idx)),
        "n_train_rows_fit": int(len(train_fit_idx)),
        "target_mean": y_mean,
        "target_std": y_std,
    }


# --- plotting ----------------------------------------------------------------------------------

def draw_candidate_panel(parent_smiles: str, source_fragment_smiles: str, candidate_fragment_smiles: str, legend_parent: str, legend_source: str, legend_candidate: str):
    parent = Chem.MolFromSmiles(parent_smiles)
    source_frag = Chem.MolFromSmiles(source_fragment_smiles)
    candidate_frag = Chem.MolFromSmiles(candidate_fragment_smiles)
    highlight_atoms = ()
    if parent is not None and source_frag is not None:
        match = parent.GetSubstructMatch(source_frag)
        if match:
            highlight_atoms = match
    return Draw.MolsToGridImage(
        [parent, source_frag, candidate_frag],
        molsPerRow=3,
        subImgSize=(300, 300),
        legends=[legend_parent, legend_source, legend_candidate],
        highlightAtomLists=[highlight_atoms, (), ()],
        useSVG=False,
    )


def save_candidate_drawings(ranked_df: pd.DataFrame, outdir: Path, draw_top_n: int) -> None:
    top_up = ranked_df.head(draw_top_n).reset_index(drop=True)
    top_down = ranked_df.tail(draw_top_n).sort_values("symbolic_pred_delta_synergy", ascending=True).reset_index(drop=True)

    def render_subset(df: pd.DataFrame, prefix: str) -> None:
        if df.empty:
            return
        panels = []
        for row in df.itertuples(index=False):
            parent_smiles = row.drug_a_smiles if row.replace_drug == "a" else row.drug_b_smiles
            panel = draw_candidate_panel(
                parent_smiles=parent_smiles,
                source_fragment_smiles=row.source_fragment_smiles,
                candidate_fragment_smiles=row.candidate_fragment_smiles,
                legend_parent=(
                    f"{row.drug_a}+{row.drug_b}\nreplace {row.replace_drug}-frag {row.fragment_index}\n"
                    f"pred dSy={row.symbolic_pred_delta_synergy:+.2f}"
                ),
                legend_source=f"source\n{row.source_fragment_smiles}",
                legend_candidate=f"candidate\n{row.candidate_fragment_smiles}\nteacher dSy={row.teacher_delta_synergy:+.2f}",
            )
            panels.append(panel)
            panel.save(outdir / f"{prefix}_candidate_{len(panels):02d}.png")

    render_subset(top_up, "top_increase")
    render_subset(top_down, "top_decrease")


def plot_split_parity(training_df: pd.DataFrame, outdir: Path) -> None:
    for split in ["train", "val", "test"]:
        df = training_df[training_df["split"] == split]
        if df.empty:
            continue
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(df["teacher_delta_synergy"], df["symbolic_pred_delta_synergy"], s=10, alpha=0.6)
        lo = float(min(df["teacher_delta_synergy"].min(), df["symbolic_pred_delta_synergy"].min()))
        hi = float(max(df["teacher_delta_synergy"].max(), df["symbolic_pred_delta_synergy"].max()))
        plt.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
        plt.xlabel("Teacher delta synergy")
        plt.ylabel("Symbolic predicted delta synergy")
        plt.title(f"{split} parity")
        plt.tight_layout()
        plt.savefig(outdir / f"{split}_parity.png", dpi=220)
        plt.close()


def plot_metric_bars(metrics: Dict[str, Dict[str, float]], outdir: Path) -> None:
    metric_names = ["rmse", "mae", "r2", "pcc"]
    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric_name in zip(axes.flatten(), metric_names):
        values = [metrics[split][metric_name] for split in splits]
        ax.bar(splits, values, color=["#4F81BD", "#C0504D", "#9BBB59"])
        ax.set_title(metric_name.upper())
    plt.tight_layout()
    plt.savefig(outdir / "split_metrics.png", dpi=220)
    plt.close(fig)


# --- main --------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    checkpoint = load_checkpoint(Path(args.checkpoint), args.device)
    model = build_model(checkpoint, args.device)
    examples = build_examples_bundle(args, checkpoint)
    fragment_embeddings = examples["fragment_embeddings"]

    top_fragments_path = resolve_top_fragments_table(args)
    if not top_fragments_path.exists():
        raise FileNotFoundError(f"Top-fragments table not found: {top_fragments_path}")
    top_fragments_df = pd.read_csv(top_fragments_path)
    if top_fragments_df.empty:
        raise ValueError(f"Top-fragments table is empty: {top_fragments_path}")

    if args.replace_drug != "both":
        top_fragments_df = top_fragments_df[top_fragments_df["drug_side"].astype(str).str.lower() == args.replace_drug]
    top_fragments_df = rank_fragment_seeds(
        top_fragments_df,
        args.seed_selection_mode,
        args.max_fragment_seeds,
        harmful_seed_threshold=args.harmful_seed_threshold,
        allow_nonnegative_harmful_seeds=args.allow_nonnegative_harmful_seeds,
    )
    if top_fragments_df.empty:
        if args.seed_selection_mode == "most_harmful" and not args.allow_nonnegative_harmful_seeds:
            raise ValueError(
                "No strictly harmful fragment seeds remain after filtering. "
                "Try relaxing --harmful-seed-threshold or use --allow-nonnegative-harmful-seeds to inspect mixed-sign seeds."
            )
        raise ValueError("No fragment seeds remain after applying the requested filters.")

    training_tables = []
    fragment_seed_summaries = []
    for seed in top_fragments_df.itertuples(index=False):
        example = pick_pair_example(examples=examples, model=model, args=args, drug_a=seed.drug_a, drug_b=seed.drug_b)
        important_fragments = pd.DataFrame([
            {
                "replace_drug": str(seed.drug_side).lower(),
                "fragment_index": int(seed.fragment_index),
                "fragment_smiles": seed.fragment_smiles,
                "attention_weight": float(getattr(seed, "attention_weight", np.nan)),
                "leave_one_out_delta": float(getattr(seed, "leave_one_out_delta", np.nan)),
                "fragment_shap_value": float(getattr(seed, "fragment_shap_value", np.nan)),
            }
        ])
        seed_training_df = build_training_table(
            model=model,
            example=example,
            important_fragments=important_fragments,
            fragment_embeddings=fragment_embeddings,
            device=args.device,
            min_similarity=args.min_morgan_similarity,
            max_heavy_atom_delta=args.max_heavy_atom_delta,
            max_candidates_per_fragment=args.max_candidates_per_fragment,
        )
        if seed_training_df.empty:
            continue
        seed_training_df["pair_rank"] = int(getattr(seed, "pair_rank", -1))
        seed_training_df["seed_fragment_abs_dim_shap"] = float(getattr(seed, "fragment_embedding_dim_abs_shap_sum", np.nan))
        training_tables.append(seed_training_df)
        fragment_seed_summaries.append(
            {
                "pair_rank": int(getattr(seed, "pair_rank", -1)),
                "drug_a": seed.drug_a,
                "drug_b": seed.drug_b,
                "drug_side": seed.drug_side,
                "fragment_index": int(seed.fragment_index),
                "fragment_smiles": seed.fragment_smiles,
                "seed_fragment_abs_dim_shap": float(getattr(seed, "fragment_embedding_dim_abs_shap_sum", np.nan)),
                "seed_selection_mode": args.seed_selection_mode,
                "n_candidates": int(len(seed_training_df)),
            }
        )

    if not training_tables:
        raise ValueError("No replacement candidates survived the filtering rules for any selected fragment seed.")
    training_df = pd.concat(training_tables, axis=0, ignore_index=True)

    fit_result = fit_symbolic_regression(training_df, args)
    training_df_filtered = fit_result["training_df_filtered"].copy()
    ranked_df = training_df_filtered.sort_values("symbolic_pred_delta_synergy", ascending=False).reset_index(drop=True)

    safe_cell = str(args.cell_line).replace("/", "_")
    outdir = Path(args.outdir) / safe_cell
    outdir.mkdir(parents=True, exist_ok=True)

    top_fragments_df.to_csv(outdir / "selected_fragment_seeds.csv", index=False)
    pd.DataFrame(fragment_seed_summaries).to_csv(outdir / "fragment_seed_summary.csv", index=False)
    training_df.to_csv(outdir / "replacement_training_data_before_filter.csv", index=False)
    training_df_filtered.to_csv(outdir / "replacement_training_data.csv", index=False)
    ranked_df.to_csv(outdir / "replacement_candidates_ranked.csv", index=False)
    fit_result["scaler_df"].to_csv(outdir / "feature_normalization.csv", index=False)
    (outdir / "symbolic_expression_raw.txt").write_text(fit_result["raw_equation"] + "\n")
    (outdir / "symbolic_expression_named.txt").write_text("delta_synergy = " + fit_result["named_equation"] + "\n")
    (outdir / "symbolic_expression_final.txt").write_text(fit_result["final_equation"])
    (outdir / "symbolic_features.json").write_text(json.dumps(fit_result["feature_columns"], indent=2))
    (outdir / "split_metrics.json").write_text(json.dumps(fit_result["metrics"], indent=2))
    pd.DataFrame([{"split": split, **metric_values} for split, metric_values in fit_result["metrics"].items()]).to_csv(
        outdir / "split_metrics.csv", index=False
    )
    plot_split_parity(training_df_filtered, outdir)
    plot_metric_bars(fit_result["metrics"], outdir)
    save_candidate_drawings(ranked_df, outdir, args.draw_top_candidates)

    summary = {
        "cell_line_name": args.cell_line,
        "top_fragments_table": str(top_fragments_path),
        "n_fragment_seeds": int(len(top_fragments_df)),
        "seed_selection_mode": args.seed_selection_mode,
        "harmful_seed_threshold": args.harmful_seed_threshold,
        "allow_nonnegative_harmful_seeds": bool(args.allow_nonnegative_harmful_seeds),
        "n_training_rows_before_filter": int(fit_result["n_rows_before_filter"]),
        "n_training_rows_after_filter": int(fit_result["n_rows_after_filter"]),
        "delta_filter_mode": args.delta_filter_mode,
        "delta_filter_threshold": args.delta_filter_threshold,
        "feature_top_k": args.feature_top_k,
        "sample_weight_power": args.sample_weight_power,
        "raw_equation": fit_result["raw_equation"],
        "named_equation": fit_result["named_equation"],
        "final_equation": fit_result["final_equation"],
        "fit_metrics": fit_result["metrics"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Cell line: {args.cell_line}")
    print(f"Fragment seeds used: {len(top_fragments_df)}")
    print(f"Seed selection mode: {args.seed_selection_mode}")
    if args.seed_selection_mode == "most_harmful":
        print(f"Harmful seed threshold: {args.harmful_seed_threshold}")
        print(f"Allow nonnegative harmful seeds: {args.allow_nonnegative_harmful_seeds}")
    print(f"Replacement training rows before filter: {fit_result['n_rows_before_filter']}")
    print(f"Replacement training rows after filter: {fit_result['n_rows_after_filter']}")
    print(f"Raw symbolic expression: {fit_result['raw_equation']}")
    print(f"Named symbolic expression: {fit_result['named_equation']}")
    print(f"Test metrics: {fit_result['metrics']['test']}")
    print("Top candidate replacements:")
    print(
        ranked_df[
            [
                "pair_rank",
                "drug_a",
                "drug_b",
                "replace_drug",
                "fragment_index",
                "source_fragment_smiles",
                "candidate_fragment_smiles",
                "symbolic_pred_delta_synergy",
                "teacher_delta_synergy",
                "morgan_tanimoto",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()