"""
E8 -- One-hot controls under the grouped split
==============================================
The grouped-split retrain (E5) showed the teacher reaches PCC 0.815 on unseen
(cell line, drug pair) combinations, well above the gradient-boosting baseline
and far above a lookup table, which has no entry to recall. That argues the
molecular representation is doing real work.

This script tests that directly by replacing representations with bare
identity codes and retraining, under the same leakage-free grouped split:

  --ablate drug   molecule and fragment embeddings -> one-hot drug identity
  --ablate cell   4,079 x 6 omics tensor           -> one-hot cell-line identity
  --ablate both   both of the above
  --ablate none   control run, identical to E5 (sanity check)

If a model given only identity codes matches the full model, the
representations are decorative. If it collapses, they are load-bearing. Under
the *random* split we expect one-hot to do well, because identity is all you
need to recall a memorised combination; the grouped split is where the question
becomes meaningful.

Nothing in the repository is modified. synergy_prediction.py is imported and
patched in memory for this process only.

Usage
-----
    python experiments/e8_onehot_controls.py --ablate drug \
        --group-mode cellpair \
        --outdir experiments/results/onehot_drug_cellpair
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

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


def norm(x):
    return str(x).strip().lower()


def group_keys(meta, mode):
    a = meta.drug_a.astype(str).str.strip().str.lower()
    b = meta.drug_b.astype(str).str.strip().str.lower()
    pair = [tuple(sorted(t)) for t in zip(a, b)]
    if mode == "pair":
        return np.asarray([str(p) for p in pair])
    cl = meta.cell_line_name.astype(str).str.strip().str.lower()
    return np.asarray(["%s|%s" % (c, p) for c, p in zip(cl, pair)])


def grouped_split(keys, val_fraction, test_fraction, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(keys)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    test_g, val_g = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
    idx = np.arange(len(keys))
    return (idx[[(k not in test_g and k not in val_g) for k in keys]],
            idx[[k in val_g for k in keys]],
            idx[[k in test_g for k in keys]])


def random_split(n, val_fraction, test_fraction, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nt = max(1, int(round(n * test_fraction)))
    nv = max(1, int(round(n * val_fraction)))
    return idx[nt + nv:], idx[nt:nt + nv], idx[:nt]


def onehot_drugs(examples):
    """Molecule and fragment embeddings -> one-hot drug identity."""
    meta = examples["meta"]
    names = sorted(set(meta.drug_a.map(norm)) | set(meta.drug_b.map(norm)))
    idx = {d: i for i, d in enumerate(names)}
    D = len(names)
    n = len(meta)

    mol_a = np.zeros((n, D), dtype=np.float32)
    mol_b = np.zeros((n, D), dtype=np.float32)
    mol_a[np.arange(n), [idx[norm(x)] for x in meta.drug_a]] = 1.0
    mol_b[np.arange(n), [idx[norm(x)] for x in meta.drug_b]] = 1.0

    examples["mol_a"], examples["mol_b"] = mol_a, mol_b
    examples["frag_a"] = [mol_a[i:i + 1].copy() for i in range(n)]
    examples["frag_b"] = [mol_b[i:i + 1].copy() for i in range(n)]
    return D


def onehot_cells(examples):
    """Omics tensor (G x 6) -> one-hot cell identity, same rank so the
    attention-pooling cell encoder is unchanged."""
    meta = examples["meta"]
    cl = sorted(set(meta.cell_line_name.map(norm)))
    idx = {c: i for i, c in enumerate(cl)}
    C = len(cl)
    n = len(meta)
    ch = int(examples["cells"].shape[-1])
    cells = np.zeros((n, C, ch), dtype=np.float32)
    cells[np.arange(n), [idx[norm(x)] for x in meta.cell_line_name], :] = 1.0
    examples["cells"] = cells
    return C


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ablate", choices=("none", "drug", "cell", "both"),
                    default="drug")
    ap.add_argument("--group-mode", choices=("cellpair", "pair", "random"),
                    default="cellpair")
    ap.add_argument("--outdir", default="experiments/results/onehot_drug_cellpair")
    mine, rest = ap.parse_known_args()

    m = load_module()
    outdir = Path(mine.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    argv = ["synergy_prediction.py"]
    for k, v in DEFAULTS.items():
        if k not in rest:
            argv += [k, v]
    argv += ["--outdir", str(outdir)] + rest
    sys.argv = argv

    audit = {"ablate": mine.ablate, "split": mine.group_mode}

    def patched_bundle(args):
        dataset = m.load_pickle(Path(args.input))
        examples = m._build_examples(
            dataset=dataset,
            molecule_embeddings=m.load_embedding_pickle(Path(args.molecule_embeddings)),
            fragment_embeddings=m.load_embedding_pickle(Path(args.fragment_embeddings)),
            fragment_map=m.load_pickle(Path(args.fragment_map)),
            drug_smiles_map=m.load_drug_smiles_map(Path(args.drug_smiles_map)),
            include_no_fragments=args.include_no_fragments,
        )

        if mine.ablate in ("drug", "both"):
            audit["n_drug_codes"] = onehot_drugs(examples)
        if mine.ablate in ("cell", "both"):
            audit["n_cell_codes"] = onehot_cells(examples)

        meta = examples["meta"]
        if mine.group_mode == "random":
            tr, va, te = random_split(len(meta), args.val_fraction,
                                      args.test_fraction, args.seed)
            leak = -1
        else:
            keys = group_keys(meta, mine.group_mode)
            tr, va, te = grouped_split(keys, args.val_fraction,
                                       args.test_fraction, args.seed)
            ktr, kva, kte = set(keys[tr]), set(keys[va]), set(keys[te])
            leak = len((ktr & kte) | (ktr & kva) | (kva & kte))
            audit["n_groups"] = int(len(set(keys)))

        audit.update(n_rows=int(len(meta)), n_train=int(len(tr)),
                     n_val=int(len(va)), n_test=int(len(te)),
                     leaked_groups=int(leak),
                     molecule_dim=int(examples["mol_a"].shape[1]),
                     fragment_dim=int(examples["frag_a"][0].shape[1]),
                     cell_shape=list(examples["cells"].shape[1:]))

        print("\n" + "=" * 70)
        print("  E8  ablate=%s  split=%s" % (mine.ablate, mine.group_mode))
        print("=" * 70)
        for k, v in audit.items():
            print("  %-16s %s" % (k, v))
        if leak > 0:
            raise SystemExit("grouped split leaked; aborting")
        print("=" * 70 + "\n")

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
        with open(outdir / "ablation_audit.json", "w") as f:
            json.dump(audit, f, indent=1)
    print("\naudit -> %s/ablation_audit.json" % outdir)


if __name__ == "__main__":
    main()
