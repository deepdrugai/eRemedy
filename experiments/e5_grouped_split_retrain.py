"""
E5 -- Retrain the teacher under a leakage-free grouped split
============================================================
E3 showed that a `groupby(['cell','pair']).mean()` lookup reaches PCC 0.969
against the teacher's reported 0.946. The cause is replication: the O'Neil
subset carries roughly 3.3 rows per (cell line, drug pair) combination, so a
random split places the same combination on both sides for ~99% of test rows.

This script retrains the *unmodified* teacher with the split replaced by a
grouped one, so that no combination appears in both train and test.

  --group-mode cellpair   groups = (cell line, unordered drug pair)
                          answers: can it predict an unseen combination?
  --group-mode pair       groups = unordered drug pair, all cell lines together
                          answers: can it predict an unseen pair anywhere?
                          (strictly harder, closest to the deployed use case)

Nothing in the repository is modified. synergy_prediction.py is loaded as a
module and its split function is replaced in memory for this process only;
all model code, hyperparameters and outputs are the originals.

Usage
-----
    python experiments/e5_grouped_split_retrain.py \
        --group-mode cellpair \
        --outdir experiments/results/teacher_grouped_cellpair
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "synergy_prediction.py"

RUN0 = REPO / "clean_runs/2026-04-29_full_rerun"
DEFAULTS = {
    "--input": "/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl",
    "--molecule-embeddings": str(RUN0 / "molecule_ae/molecule_smiles_embeddings.pkl"),
    "--fragment-embeddings": str(RUN0 / "fragment_ae/fragment_smiles_embeddings.pkl"),
    "--fragment-map": str(RUN0 / "fragment_ae/smiles_to_fragments.pkl"),
    "--drug-smiles-map": str(REPO / "everything_drug_smiles_map.csv"),
}


def load_module():
    spec = importlib.util.spec_from_file_location("sp_teacher", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def group_keys(meta: pd.DataFrame, mode: str):
    a = meta.drug_a.astype(str).str.strip().str.lower()
    b = meta.drug_b.astype(str).str.strip().str.lower()
    pair = [tuple(sorted(t)) for t in zip(a, b)]
    if mode == "pair":
        return np.asarray([str(p) for p in pair])
    cl = meta.cell_line_name.astype(str).str.strip().str.lower()
    return np.asarray(["%s|%s" % (c, p) for c, p in zip(cl, pair)])


def grouped_split(keys, val_fraction, test_fraction, seed):
    """Assign whole groups to train/val/test; no group spans two splits."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(keys)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    test_g = set(uniq[:n_test])
    val_g = set(uniq[n_test:n_test + n_val])
    idx = np.arange(len(keys))
    te = idx[[k in test_g for k in keys]]
    va = idx[[k in val_g for k in keys]]
    tr = idx[[(k not in test_g and k not in val_g) for k in keys]]
    return tr, va, te


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--group-mode", choices=("cellpair", "pair"), default="cellpair")
    ap.add_argument("--outdir", default="experiments/results/teacher_grouped_cellpair")
    mine, rest = ap.parse_known_args()

    m = load_module()
    outdir = Path(mine.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # feed synergy_prediction its own CLI, with our paths unless overridden
    argv = ["synergy_prediction.py"]
    for k, v in DEFAULTS.items():
        if k not in rest:
            argv += [k, v]
    argv += ["--outdir", str(outdir)] + rest
    sys.argv = argv

    original_bundle = m._build_split_bundle
    audit = {}

    def patched_bundle(args):
        dataset = m.load_pickle(Path(args.input))
        molecule_embeddings = m.load_embedding_pickle(Path(args.molecule_embeddings))
        fragment_embeddings = m.load_embedding_pickle(Path(args.fragment_embeddings))
        fragment_map = m.load_pickle(Path(args.fragment_map))
        drug_smiles_map = m.load_drug_smiles_map(Path(args.drug_smiles_map))

        examples = m._build_examples(
            dataset=dataset,
            molecule_embeddings=molecule_embeddings,
            fragment_embeddings=fragment_embeddings,
            fragment_map=fragment_map,
            drug_smiles_map=drug_smiles_map,
            include_no_fragments=args.include_no_fragments,
        )

        meta = examples["meta"]
        keys = group_keys(meta, mine.group_mode)
        tr, va, te = grouped_split(keys, args.val_fraction,
                                   args.test_fraction, args.seed)

        ktr, kva, kte = set(keys[tr]), set(keys[va]), set(keys[te])
        leak = len((ktr & kte) | (ktr & kva) | (kva & kte))
        audit.update(dict(
            group_mode=mine.group_mode,
            n_rows=int(len(keys)), n_groups=int(len(set(keys))),
            rows_per_group=float(len(keys) / len(set(keys))),
            n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
            groups_train=len(ktr), groups_val=len(kva), groups_test=len(kte),
            leaked_groups=int(leak)))
        print("\n" + "=" * 70)
        print("  GROUPED SPLIT (%s)" % mine.group_mode)
        print("=" * 70)
        for k, v in audit.items():
            print("  %-16s %s" % (k, v))
        print("  leaked groups MUST be 0 -> %s" % ("OK" if leak == 0 else "FAIL"))
        print("=" * 70 + "\n")
        if leak:
            raise SystemExit("grouped split leaked; aborting")

        return {
            "train": m._slice_examples(examples, tr),
            "val": m._slice_examples(examples, va),
            "test": m._slice_examples(examples, te),
            "dataset_total": examples["total_rows"],
            "dataset_valid": examples["valid_rows"],
            "dataset_skipped": examples["skipped"],
            "molecule_dim": int(examples["mol_a"].shape[1]),
            "fragment_dim": int(examples["frag_a"][0].shape[1]),
        }

    m._build_split_bundle = patched_bundle
    try:
        m.main()
    finally:
        m._build_split_bundle = original_bundle
        with open(outdir / "grouped_split_audit.json", "w") as f:
            json.dump(audit, f, indent=1)
    print("\nsplit audit -> %s/grouped_split_audit.json" % outdir)


if __name__ == "__main__":
    main()
