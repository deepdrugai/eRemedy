"""
E11 (stage 0) -- ECFP molecule and fragment embeddings
======================================================
E8 showed that replacing the teacher's molecule and fragment embeddings with a
21-way one-hot drug code costs nothing and slightly helps, while E10 showed
DeepSynergy *loses* 0.11-0.15 PCC when its ECFP features are replaced by the
same one-hot code. Chemistry is therefore learnable on O'Neil; the frozen
SMILES-autoencoder embeddings are what fail to carry it.

This script rebuilds both embedding pickles with ECFP6 + RDKit descriptors,
the featurisation DeepSynergy uses, in exactly the format
`synergy_prediction.load_embedding_pickle` expects:

    dict[str, np.ndarray]   SMILES -> float32 vector

Keys are copied verbatim from the existing pickles (plus anything reachable
through the fragment map and the drug SMILES map), so downstream lookups
resolve identically and coverage cannot silently regress. Nothing else in the
architecture changes: the fragment channel stays fragment-resolved, one vector
per fragment, so fragment SHAP, leave-one-out ranking and candidate swap
rescoring all behave as before.

Descriptors are standardised then squashed with tanh, as in DeepSynergy;
without this the raw descriptor scales (MolWt ~500 against 0/1 bits) would
dominate the input.

Usage
-----
    python experiments/e11_build_ecfp_embeddings.py \
        --outdir clean_runs/2026-08-28_ecfp_rerun/ecfp
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[1]
RUN0 = REPO / "clean_runs/2026-04-29_full_rerun"


def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def ecfp(smiles, n_bits, radius, with_descriptors):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    bits = np.zeros(n_bits, dtype=np.float32)
    for b in fp.GetOnBits():
        bits[b] = 1.0
    if not with_descriptors:
        return bits
    desc = []
    for _, fn in Descriptors._descList:
        try:
            v = float(fn(mol))
        except Exception:
            v = 0.0
        desc.append(v if np.isfinite(v) else 0.0)
    # A few descriptors (Ipc especially) are finite in float64 but overflow
    # float32 to inf, which poisons the standardisation downstream.
    d = np.clip(np.asarray(desc, dtype=np.float64), -1e30, 1e30)
    return np.concatenate([bits.astype(np.float64), d])


def build(keys, n_bits, radius, with_descriptors, label):
    """Featurise every key, tanh-normalising the descriptor block."""
    keys = sorted(keys)
    vecs, failed = {}, []
    for k in keys:
        v = ecfp(k, n_bits, radius, with_descriptors)
        if v is None:
            failed.append(k)
        else:
            vecs[k] = v

    if not vecs:
        raise SystemExit("%s: every SMILES failed to parse" % label)

    dim = len(next(iter(vecs.values())))
    M = np.stack([vecs[k] for k in vecs.keys()]).astype(np.float64)

    if with_descriptors:
        d = M[:, n_bits:]
        mu, sd = d.mean(0), d.std(0)
        sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
        M[:, n_bits:] = np.tanh((d - mu) / sd)

    M = np.nan_to_num(M, nan=0.0, posinf=1.0, neginf=-1.0)
    if not np.isfinite(M).all():
        raise SystemExit("%s: non-finite values survived normalisation" % label)

    out = {k: M[i].astype(np.float32) for i, k in enumerate(vecs.keys())}
    for k in failed:
        out[k] = np.zeros(dim, dtype=np.float32)

    print("  %-9s %6d keys -> %d-d  (%d unparseable, zero-filled)"
          % (label, len(out), dim, len(failed)))
    return out, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule-embeddings",
                    default=str(RUN0 / "molecule_ae/molecule_smiles_embeddings.pkl"),
                    help="existing pickle, read only to copy its key set")
    ap.add_argument("--fragment-embeddings",
                    default=str(RUN0 / "fragment_ae/fragment_smiles_embeddings.pkl"))
    ap.add_argument("--fragment-map",
                    default=str(RUN0 / "fragment_ae/smiles_to_fragments.pkl"))
    ap.add_argument("--drug-smiles-map",
                    default=str(REPO / "everything_drug_smiles_map.csv"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-bits", type=int, default=1024)
    ap.add_argument("--radius", type=int, default=3, help="3 => ECFP6")
    ap.add_argument("--no-descriptors", action="store_true")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with_desc = not a.no_descriptors

    mol_old = load_pickle(a.molecule_embeddings)
    frag_old = load_pickle(a.fragment_embeddings)
    frag_map = load_pickle(a.fragment_map)

    mol_keys = set(mol_old)
    mol_keys |= set(frag_map.keys())
    import csv
    with open(a.drug_smiles_map, newline="") as f:
        for row in csv.DictReader(f):
            for col in ("smiles", "SMILES", "canonical_smiles"):
                if row.get(col):
                    mol_keys.add(str(row[col]).strip())
                    break

    frag_keys = set(frag_old)
    for v in frag_map.values():
        frag_keys |= {str(s).strip() for s in (v or [])}

    print("=" * 70)
    print("  E11  ECFP%d embeddings  (%d bits%s)"
          % (a.radius * 2, a.n_bits, " + descriptors" if with_desc else ""))
    print("=" * 70)
    print("  source key sets: molecule %d (was %d), fragment %d (was %d)"
          % (len(mol_keys), len(mol_old), len(frag_keys), len(frag_old)))

    mol_new, mol_dim = build(mol_keys, a.n_bits, a.radius, with_desc, "molecule")
    frag_new, frag_dim = build(frag_keys, a.n_bits, a.radius, with_desc, "fragment")

    missing_mol = [k for k in mol_old if k not in mol_new]
    missing_frag = [k for k in frag_old if k not in frag_new]
    if missing_mol or missing_frag:
        raise SystemExit("coverage regression: %d molecule, %d fragment keys lost"
                         % (len(missing_mol), len(missing_frag)))

    mp = out / "molecule_smiles_embeddings.pkl"
    fp = out / "fragment_smiles_embeddings.pkl"
    with open(mp, "wb") as f:
        pickle.dump(mol_new, f)
    with open(fp, "wb") as f:
        pickle.dump(frag_new, f)

    import json
    json.dump(dict(n_bits=a.n_bits, radius=a.radius, descriptors=with_desc,
                   molecule_keys=len(mol_new), molecule_dim=mol_dim,
                   fragment_keys=len(frag_new), fragment_dim=frag_dim,
                   featuriser="ECFP%d + RDKit descriptors (tanh-normalised)"
                              % (a.radius * 2) if with_desc else
                              "ECFP%d bits" % (a.radius * 2)),
              open(out / "ecfp_audit.json", "w"), indent=1)

    print("\n  every original key covered")
    print("  -> %s" % mp)
    print("  -> %s" % fp)


if __name__ == "__main__":
    main()
