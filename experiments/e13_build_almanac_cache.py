"""
E13 (stage 0) -- compact ALMANAC cache for the vocabulary-scaling experiment
============================================================================
`dataset_ALMANAC.pkl` is 23.9 GB, almost all of which is a per-row copy of the
cell-line omics tensor. There are only a few dozen distinct cell lines, so the
same information collapses to a few megabytes once it is stored per cell line
and indexed. Loading the full pickle for every run in a scaling sweep would be
prohibitive; loading it once is not.

Writes:
    meta.pkl        DataFrame: drug_a, drug_b, cell_line, cell_idx, y
    omics.npy       (n_cells, G, 6) float32, one row per distinct cell line
    vocab.json      encodable drugs, their SMILES and fragment lists

A drug counts as encodable on the same terms the teacher uses: it resolves to a
SMILES through the drug map and decomposes into at least one fragment.

Usage
-----
    python experiments/e13_build_almanac_cache.py \
        --outdir clean_runs/almanac_cache
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN0 = REPO / "clean_runs/2026-04-29_full_rerun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/media/exxact1/magesh/ecomb/dataset_ALMANAC.pkl")
    ap.add_argument("--fragment-map", default=str(RUN0 / "fragment_ae/smiles_to_fragments.pkl"))
    ap.add_argument("--drug-smiles-map", default=str(REPO / "everything_drug_smiles_map.csv"))
    ap.add_argument("--molecule-embeddings",
                    default=str(RUN0 / "molecule_ae/molecule_smiles_embeddings.pkl"))
    ap.add_argument("--target", default="y_loewe")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("sp", REPO / "synergy_prediction.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)

    frag_map = sp.load_pickle(Path(a.fragment_map))
    drug_map = sp.load_drug_smiles_map(Path(a.drug_smiles_map))
    low = {str(k).strip().lower(): v for k, v in drug_map.items()}
    mol_emb = sp.load_embedding_pickle(Path(a.molecule_embeddings))

    print("loading ALMANAC (this is the slow part) -> %s" % a.dataset, flush=True)
    ds = sp.load_pickle(Path(a.dataset))
    meta = ds["meta"]
    omics = ds["cell_omics"]
    print("  rows: %s" % format(len(meta), ","), flush=True)

    da = meta.drug_a.astype(str).str.strip()
    db = meta.drug_b.astype(str).str.strip()
    cl = meta.cell_line_name.astype(str).str.strip()

    # ---- encodable vocabulary ------------------------------------------
    vocab = {}
    for d in sorted(set(da) | set(db)):
        s = low.get(d.lower())
        if not s:
            continue
        frags = [str(x).strip() for x in (frag_map.get(s) or [])]
        if not frags:
            continue
        vocab[d] = {"smiles": s, "fragments": frags,
                    "has_ae_embedding": bool(mol_emb.get(s) is not None)}
    n_ae = sum(v["has_ae_embedding"] for v in vocab.values())
    print("  encodable drugs: %d (%d also have an autoencoder embedding)"
          % (len(vocab), n_ae), flush=True)

    keep = da.isin(vocab) & db.isin(vocab)
    y = pd.to_numeric(meta[a.target], errors="coerce")
    keep &= y.notna()
    print("  rows with both drugs encodable and a target: %s"
          % format(int(keep.sum()), ","), flush=True)

    # ---- collapse the omics to one row per cell line --------------------
    cells = sorted(set(cl[keep]))
    c_idx = {c: i for i, c in enumerate(cells)}
    first = {}
    for i, c in enumerate(cl):
        if keep.iat[i] and c not in first:
            first[c] = i
    O = np.stack([np.asarray(omics[first[c]], dtype=np.float32) for c in cells])
    print("  cell lines: %d, omics tensor %s (%.1f MB)"
          % (len(cells), O.shape, O.nbytes / 1e6), flush=True)

    m = pd.DataFrame({
        "drug_a": da[keep].values,
        "drug_b": db[keep].values,
        "cell_line": cl[keep].values,
        "y": y[keep].astype(np.float32).values,
    })
    m["cell_idx"] = m.cell_line.map(c_idx).astype(np.int32)

    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(m, f)
    np.save(out / "omics.npy", O)
    json.dump({"cells": cells, "vocab": vocab,
               "n_rows": int(len(m)), "target": a.target},
              open(out / "vocab.json", "w"), indent=1)

    per_drug = pd.concat([m.drug_a, m.drug_b]).value_counts()
    print("\n  rows per drug: median %d, min %d, max %d"
          % (per_drug.median(), per_drug.min(), per_drug.max()))
    print("  cache -> %s" % out)


if __name__ == "__main__":
    main()
