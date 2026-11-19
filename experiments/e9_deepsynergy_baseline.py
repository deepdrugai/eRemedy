"""
E9 -- DeepSynergy baseline under matched splits
===============================================
A reimplementation of DeepSynergy (Preuer et al., Bioinformatics 2018) trained
on exactly the same rows and exactly the same splits as our teacher, so the
comparison isolates the model rather than the data.

Fidelity to the original
------------------------
  kept      concatenated [drug A | drug B | cell] flat feature vector
  kept      ECFP6 (radius 3, 1024 bits) plus physicochemical descriptors
  kept      two large hidden layers (8192, 4096), ReLU, dropout 0.5
  kept      tanh input normalisation, MSE loss
  kept      symmetry augmentation: each pair is presented in both orderings

  adapted   cell features are the omics tensor this study uses, flattened and
            reduced to its highest-variance columns, rather than the original
            3,984-gene expression vector. Dimensionality is kept comparable.
  adapted   toxicophore descriptors are omitted; they came from a tool we do
            not have. ECFP6 plus 208 RDKit descriptors stands in.
  adapted   Adam rather than the original SGD at lr 1e-5. Adam converges in far
            fewer epochs and reaches equal or better validation loss, so this
            strengthens rather than handicaps the baseline.

Matched evaluation
------------------
Rows come from synergy_prediction._build_examples, so the row set is identical
to the teacher's 23,872 (a row is dropped if either drug lacks a fragment
decomposition). Splits use the same grouping rule and the same seed as E5/E8.

Usage
-----
    python experiments/e9_deepsynergy_baseline.py --split cellpair \
        --outdir experiments/results/deepsynergy_cellpair
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from torch.utils.data import DataLoader, Dataset

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "synergy_prediction.py"
RUN0 = REPO / "clean_runs/2026-04-29_full_rerun"

DEFAULTS = dict(
    input="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl",
    molecule_embeddings=str(RUN0 / "molecule_ae/molecule_smiles_embeddings.pkl"),
    fragment_embeddings=str(RUN0 / "fragment_ae/fragment_smiles_embeddings.pkl"),
    fragment_map=str(RUN0 / "fragment_ae/smiles_to_fragments.pkl"),
    drug_smiles_map=str(REPO / "everything_drug_smiles_map.csv"),
)


def norm(x):
    return str(x).strip().lower()


# ---------------------------------------------------------------- features --
def drug_features(smiles, n_bits=1024, radius=3):
    """ECFP6 plus RDKit physicochemical descriptors, as in the original."""
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for b in fp.GetOnBits():
        arr[b] = 1.0
    desc = []
    for _, fn in Descriptors._descList:
        try:
            v = fn(m)
        except Exception:
            v = 0.0
        desc.append(v if np.isfinite(v) else 0.0)
    return np.concatenate([arr, np.asarray(desc, dtype=np.float32)])


def tanh_norm(X, mu=None, sd=None):
    """DeepSynergy's tanh normalisation: standardise, then squash."""
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-8] = 1.0
    return np.tanh((X - mu) / sd).astype(np.float32), mu, sd


# ------------------------------------------------------------------ splits --
def group_keys(meta, mode):
    a = meta.drug_a.map(norm)
    b = meta.drug_b.map(norm)
    pair = [tuple(sorted(t)) for t in zip(a, b)]
    if mode == "pair":
        return np.asarray([str(p) for p in pair])
    cl = meta.cell_line_name.map(norm)
    return np.asarray(["%s|%s" % (c, p) for c, p in zip(cl, pair)])


def make_split(meta, mode, seed, val=0.1, test=0.1):
    n = len(meta)
    rng = np.random.default_rng(seed)
    if mode == "random":
        idx = rng.permutation(n)
        nt, nv = max(1, int(round(n * test))), max(1, int(round(n * val)))
        return idx[nt + nv:], idx[nt:nt + nv], idx[:nt], -1
    keys = group_keys(meta, mode)
    uniq = np.unique(keys)
    rng.shuffle(uniq)
    g = len(uniq)
    nt, nv = max(1, int(round(g * test))), max(1, int(round(g * val)))
    te_g, va_g = set(uniq[:nt]), set(uniq[nt:nt + nv])
    idx = np.arange(n)
    tr = idx[[(k not in te_g and k not in va_g) for k in keys]]
    va = idx[[k in va_g for k in keys]]
    te = idx[[k in te_g for k in keys]]
    leak = len((set(keys[tr]) & set(keys[te])) |
               (set(keys[tr]) & set(keys[va])) |
               (set(keys[va]) & set(keys[te])))
    return tr, va, te, leak


# ------------------------------------------------------------------- model --
class DeepSynergy(nn.Module):
    def __init__(self, d_in, hidden=(8192, 4096), p_in=0.2, p_hidden=0.5):
        super().__init__()
        layers, prev = [nn.Dropout(p_in)], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_hidden)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Rows(Dataset):
    """Assembles [drug A | drug B | cell] on the fly from small lookup tables."""

    def __init__(self, di_a, di_b, ci, y, DF, CF, symmetrise=False):
        self.a, self.b, self.c, self.y = di_a, di_b, ci, y
        self.DF, self.CF, self.sym = DF, CF, symmetrise

    def __len__(self):
        return len(self.y) * (2 if self.sym else 1)

    def __getitem__(self, i):
        n = len(self.y)
        j, flip = (i % n, i >= n) if self.sym else (i, False)
        a, b = (self.b[j], self.a[j]) if flip else (self.a[j], self.b[j])
        x = np.concatenate([self.DF[a], self.DF[b], self.CF[self.c[j]]])
        return torch.from_numpy(x), torch.tensor(self.y[j], dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            P.append(model(x.to(device)).cpu().numpy())
            Y.append(y.numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return dict(count=int(len(y)),
                rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                mae=float(np.mean(np.abs(y - p))),
                pearson=float(np.corrcoef(y, p)[0, 1]),
                spearman=float(__import__("scipy.stats", fromlist=["x"])
                               .spearmanr(y, p).correlation)), p, y


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("random", "cellpair", "pair"),
                    default="cellpair")
    ap.add_argument("--outdir", default="experiments/results/deepsynergy_cellpair")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--n-cell-features", type=int, default=4000)
    ap.add_argument("--drug-features", choices=("ecfp", "onehot"),
                    default="ecfp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # ---- identical rows to the teacher ----------------------------------
    spec = importlib.util.spec_from_file_location("sp_teacher", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    print("building the teacher's exact row set ...")
    ex = m._build_examples(
        dataset=m.load_pickle(Path(DEFAULTS["input"])),
        molecule_embeddings=m.load_embedding_pickle(Path(DEFAULTS["molecule_embeddings"])),
        fragment_embeddings=m.load_embedding_pickle(Path(DEFAULTS["fragment_embeddings"])),
        fragment_map=m.load_pickle(Path(DEFAULTS["fragment_map"])),
        drug_smiles_map=m.load_drug_smiles_map(Path(DEFAULTS["drug_smiles_map"])),
        include_no_fragments=False,
    )
    meta = ex["meta"].reset_index(drop=True)
    y = ex["targets"].astype(np.float32)
    print("rows: %s (matches teacher)" % format(len(meta), ","))

    # ---- cell features: flatten omics, keep highest-variance columns -----
    cl_names = sorted(set(meta.cell_line_name.map(norm)))
    cl_idx = {c: i for i, c in enumerate(cl_names)}
    first = {}
    for i, c in enumerate(meta.cell_line_name.map(norm)):
        first.setdefault(c, i)
    CF_full = np.stack([ex["cells"][first[c]].reshape(-1) for c in cl_names])
    var = CF_full.var(axis=0)
    keep = np.argsort(var)[::-1][:a.n_cell_features]
    CF = CF_full[:, keep].astype(np.float32)
    print("cell features: %d of %d columns kept by variance"
          % (CF.shape[1], CF_full.shape[1]))
    del CF_full, ex

    # ---- drug features ---------------------------------------------------
    smi = {}
    for n_, s_ in zip(meta.drug_a, meta.drug_a_smiles):
        smi.setdefault(norm(n_), s_)
    for n_, s_ in zip(meta.drug_b, meta.drug_b_smiles):
        smi.setdefault(norm(n_), s_)
    d_names = sorted(smi)
    d_idx = {d: i for i, d in enumerate(d_names)}
    if a.drug_features == "onehot":
        DF = np.eye(len(d_names), dtype=np.float32)
        print("drug features: %d drugs x %d (one-hot identity)" % DF.shape)
    else:
        feats = [drug_features(smi[d]) for d in d_names]
        dim = max(f.shape[0] for f in feats if f is not None)
        DF = np.stack([f if f is not None else np.zeros(dim, np.float32)
                       for f in feats]).astype(np.float32)
        print("drug features: %d drugs x %d (ECFP6 + descriptors)" % DF.shape)

    di_a = np.asarray([d_idx[norm(x)] for x in meta.drug_a])
    di_b = np.asarray([d_idx[norm(x)] for x in meta.drug_b])
    ci = np.asarray([cl_idx[norm(x)] for x in meta.cell_line_name])

    # ---- split -----------------------------------------------------------
    tr, va, te, leak = make_split(meta, a.split, a.seed)
    print("\n" + "=" * 68)
    print("  DeepSynergy | split=%s" % a.split)
    print("=" * 68)
    print("  train %d | val %d | test %d | leaked groups %s"
          % (len(tr), len(va), len(te), leak))
    if leak > 0:
        raise SystemExit("split leaked; aborting")

    # ---- normalise using training statistics only ------------------------
    DF_tr = np.concatenate([DF[di_a[tr]], DF[di_b[tr]]])
    _, mu_d, sd_d = tanh_norm(DF_tr)
    DF, _, _ = tanh_norm(DF, mu_d, sd_d)
    _, mu_c, sd_c = tanh_norm(CF[ci[tr]])
    CF, _, _ = tanh_norm(CF, mu_c, sd_c)

    d_in = 2 * DF.shape[1] + CF.shape[1]
    print("  input dimension %d (DeepSynergy original: 6,602)" % d_in)
    print("=" * 68 + "\n")

    def loader(idx, sym, shuffle):
        return DataLoader(Rows(di_a[idx], di_b[idx], ci[idx], y[idx], DF, CF, sym),
                          batch_size=a.batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True, drop_last=False)

    tl, vl, sl = loader(tr, True, True), loader(va, False, False), loader(te, False, False)

    model = DeepSynergy(d_in).to(a.device)
    print("parameters: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.MSELoss()

    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, yy in tl:
            opt.zero_grad()
            l = lossf(model(x.to(a.device)), yy.to(a.device))
            l.backward()
            opt.step()
            tot += l.item()
            nb += 1
        vm, _, _ = evaluate(model, vl, a.device)
        hist.append(dict(epoch=ep, train_mse=tot / nb, **{"val_" + k: v
                                                          for k, v in vm.items()}))
        print("Epoch %03d | train MSE=%.4f | val RMSE=%.4f | val Pearson=%.4f | %.1fs"
              % (ep, tot / nb, vm["rmse"], vm["pearson"], time.time() - t0),
              flush=True)
        if vm["rmse"] < best - 1e-4:
            best, bad = vm["rmse"], 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping at epoch %d (best val RMSE %.4f)" % (ep, best))
                break

    if best_state:
        model.load_state_dict(best_state)

    res = {}
    for name, ld in (("train", loader(tr, False, False)), ("val", vl), ("test", sl)):
        mm, p, yy = evaluate(model, ld, a.device)
        res["metrics_" + name] = mm
        np.savetxt(out / ("%s_predictions.csv" % name),
                   np.c_[yy, p], delimiter=",", header="y_true,y_pred",
                   comments="")
        print("%-5s RMSE=%.4f MAE=%.4f Pearson=%.4f Spearman=%.4f"
              % (name, mm["rmse"], mm["mae"], mm["pearson"], mm["spearman"]))

    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    json.dump(dict(split=a.split, leaked_groups=int(leak), d_in=int(d_in),
                   n_rows=int(len(meta)), n_train=int(len(tr)),
                   n_val=int(len(va)), n_test=int(len(te)),
                   drug_features=a.drug_features,
                   drug_feature_dim=int(DF.shape[1]),
                   cell_feature_dim=int(CF.shape[1]),
                   epochs_run=len(hist), best_val_rmse=float(best),
                   model="DeepSynergy (Preuer et al. 2018), reimplemented"),
              open(out / "run_audit.json", "w"), indent=1)
    import pandas as pd
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    torch.save(model.state_dict(), out / "model.pt")
    print("\nwritten -> %s" % out)


if __name__ == "__main__":
    main()
