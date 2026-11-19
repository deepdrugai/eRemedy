"""
validate_lodo.py
================
Leave-One-Drug-Out (LODO) validation of the synergy teacher model.

WHY THIS EXISTS
---------------
The headline teacher metric (test PCC 0.946) comes from a *random* split over
samples (`_split_indices` shuffles rows). Because the O'Neil panel contains only
~21 usable drugs measured across 34 cell lines, every drug in the test set also
appears in the training set. That metric therefore measures generalisation over
(pair x cell line) combinations of a KNOWN drug vocabulary.

The repurposing flow needs something different: it scores drugs the model has
NEVER seen (irinotecan, XL147, AZD2014, ...). LODO measures exactly that
capability -- hold out every combination involving one drug, train on the rest,
and predict the held-out drug's combinations.

If LODO PCC is high, scoring unseen drugs is defensible. If it is low, the
drug-level repurposing numbers cannot be trusted.

USAGE
-----
  python validate_lodo.py --outdir clean_runs/lodo --epochs 200
  python validate_lodo.py --drugs Veliparib metformin --epochs 50   # quick test
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import synergy_prediction as sp

DEFAULTS = {
    "input": "/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl",
    "molecule_embeddings": "clean_runs/2026-04-29_full_rerun/molecule_ae/molecule_smiles_embeddings.pkl",
    "fragment_embeddings": "clean_runs/2026-04-29_full_rerun/fragment_ae/fragment_smiles_embeddings.pkl",
    "fragment_map": "clean_runs/2026-04-29_full_rerun/fragment_ae/smiles_to_fragments.pkl",
    "drug_smiles_map": "everything_drug_smiles_map.csv",
}


def parse_args():
    p = argparse.ArgumentParser(description="Leave-one-drug-out validation")
    p.add_argument("--input", default=DEFAULTS["input"])
    p.add_argument("--molecule-embeddings", default=DEFAULTS["molecule_embeddings"])
    p.add_argument("--fragment-embeddings", default=DEFAULTS["fragment_embeddings"])
    p.add_argument("--fragment-map", default=DEFAULTS["fragment_map"])
    p.add_argument("--drug-smiles-map", default=DEFAULTS["drug_smiles_map"])
    p.add_argument("--outdir", default="clean_runs/lodo")
    p.add_argument("--drugs", nargs="*", default=None,
                   help="Subset of drugs to hold out (default: all)")
    p.add_argument("--min-test-rows", type=int, default=50,
                   help="Skip drugs with fewer than this many held-out rows")
    # training hyperparameters -- keep identical to the original teacher run
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-sym", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--d-cell", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-no-fragments", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_loader(examples, indices, args, shuffle):
    cells, mol_a, mol_b, frag_a, frag_b, y, meta, _, n, _ = sp._slice_examples(
        examples, indices)
    ds = sp.SynergyDataset(cells, mol_a, mol_b, frag_a, frag_b, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                        drop_last=False, collate_fn=sp.collate_batch)
    return loader, meta, n


def main():
    args = parse_args()
    sp.set_seed(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Building examples (once) ...")
    dataset = sp.load_pickle(Path(args.input))
    mol_emb = sp.load_embedding_pickle(Path(args.molecule_embeddings))
    frag_emb = sp.load_embedding_pickle(Path(args.fragment_embeddings))
    frag_map = sp.load_pickle(Path(args.fragment_map))
    drug_map = sp.load_drug_smiles_map(Path(args.drug_smiles_map))

    examples = sp._build_examples(
        dataset=dataset,
        molecule_embeddings=mol_emb,
        fragment_embeddings=frag_emb,
        fragment_map=frag_map,
        drug_smiles_map=drug_map,
        include_no_fragments=args.include_no_fragments,
    )
    meta_all = examples["meta"]
    n_all = examples["valid_rows"]
    print("usable rows: %d" % n_all)

    drugs = sorted(set(meta_all["drug_a"].astype(str))
                   | set(meta_all["drug_b"].astype(str)))
    if args.drugs:
        drugs = [d for d in drugs if d in set(args.drugs)]
    print("drugs to hold out: %d" % len(drugs))

    a = meta_all["drug_a"].astype(str).values
    b = meta_all["drug_b"].astype(str).values

    rows, pooled_true, pooled_pred, pooled_drug = [], [], [], []

    for i, drug in enumerate(drugs, 1):
        held = np.where((a == drug) | (b == drug))[0]
        if len(held) < args.min_test_rows:
            print("[%2d/%d] skip %-34s (only %d rows)"
                  % (i, len(drugs), drug, len(held)))
            continue

        rest = np.setdiff1d(np.arange(n_all), held)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(rest)
        n_val = max(1, int(round(len(rest) * args.val_fraction)))
        val_idx, train_idx = rest[:n_val], rest[n_val:]

        print("[%2d/%d] hold out %-34s test=%5d train=%5d"
              % (i, len(drugs), drug, len(held), len(train_idx)), flush=True)

        train_loader, _, _ = make_loader(examples, train_idx, args, True)
        val_loader, _, _ = make_loader(examples, val_idx, args, False)
        test_loader, test_meta, _ = make_loader(examples, held, args, False)

        sp.set_seed(args.seed)
        model = sp.SynergyModel(
            d_graph=int(examples["mol_a"].shape[1]),
            d_frag=int(examples["frag_a"][0].shape[1]),
            d_cell=args.d_cell,
            hidden=args.hidden,
            dropout=args.dropout,
        )
        model, _ = sp.train_one(model, train_loader, val_loader, args)

        metrics, y_true, y_pred = sp.evaluate(
            model, test_loader, args.device, return_arrays=True)

        rows.append({
            "drug": drug,
            "n_test": len(held),
            "n_train": len(train_idx),
            "pearson": metrics.get("pearson"),
            "spearman": metrics.get("spearman"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "y_true_mean": float(np.mean(y_true)),
            "y_pred_mean": float(np.mean(y_pred)),
        })
        pooled_true.append(np.asarray(y_true))
        pooled_pred.append(np.asarray(y_pred))
        pooled_drug.append(np.repeat(drug, len(y_true)))

        pd.DataFrame({"y_true": y_true, "y_pred": y_pred,
                      "cell_line": test_meta["cell_line_name"].values,
                      "drug_a": test_meta["drug_a"].values,
                      "drug_b": test_meta["drug_b"].values}).to_csv(
            out / ("predictions_%s.csv" % drug.replace("/", "_")), index=False)

        print("        -> PCC=%.3f  RMSE=%.2f  MAE=%.2f"
              % (metrics.get("pearson", float("nan")),
                 metrics.get("rmse", float("nan")),
                 metrics.get("mae", float("nan"))), flush=True)

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("pearson", ascending=False)
    df.to_csv(out / "lodo_per_drug.csv", index=False)

    yt = np.concatenate(pooled_true)
    yp = np.concatenate(pooled_pred)
    pd.DataFrame({"y_true": yt, "y_pred": yp,
                  "drug": np.concatenate(pooled_drug)}).to_csv(
        out / "lodo_pooled_predictions.csv", index=False)

    pooled = {
        "n": int(len(yt)),
        "pearson": float(np.corrcoef(yt, yp)[0, 1]),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "mae": float(np.mean(np.abs(yt - yp))),
        "n_drugs": int(len(df)),
        "mean_per_drug_pearson": float(df.pearson.mean()),
        "median_per_drug_pearson": float(df.pearson.median()),
        "min_per_drug_pearson": float(df.pearson.min()),
        "max_per_drug_pearson": float(df.pearson.max()),
        "random_split_reference_pearson": 0.9463180736262481,
    }
    (out / "lodo_summary.json").write_text(json.dumps(pooled, indent=2))

    print("\n" + "=" * 74)
    print("  LEAVE-ONE-DRUG-OUT RESULTS")
    print("=" * 74)
    print(df.to_string(index=False))
    print("-" * 74)
    print("  pooled PCC              : %.4f" % pooled["pearson"])
    print("  pooled RMSE             : %.3f" % pooled["rmse"])
    print("  mean per-drug PCC       : %.4f" % pooled["mean_per_drug_pearson"])
    print("  median per-drug PCC     : %.4f" % pooled["median_per_drug_pearson"])
    print("  range                   : %.3f .. %.3f"
          % (pooled["min_per_drug_pearson"], pooled["max_per_drug_pearson"]))
    print("  random-split reference  : %.4f" % pooled["random_split_reference_pearson"])
    print("  DEGRADATION             : %.4f"
          % (pooled["random_split_reference_pearson"] - pooled["pearson"]))
    print("=" * 74)
    print("\nWritten to", out)


if __name__ == "__main__":
    main()
