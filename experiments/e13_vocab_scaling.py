"""
E13 -- does representation learning have a vocabulary threshold?
================================================================
On O'Neil's 21 drugs every learned molecular representation we tried loses to a
bare one-hot drug code, and the end-to-end encoder (E12) overfits badly
(train 0.94, test 0.78). The natural explanation is that 21 molecules is simply
not enough to learn a molecular encoder from: there is nothing to generalise
across, so a learned encoder memorises and a fixed featurisation does not.

That is a claim about a *threshold*, and a threshold claim needs both sides of
it. NCI-ALMANAC supplies the other side, with substantially more encodable
drugs than O'Neil.

This script trains one arm at one vocabulary size:

    --arm learned   end-to-end GCN encoders (the E12 architecture)
    --arm onehot    one-hot drug identity, the control that has to be beaten
    --arm ecfp      ECFP6 + descriptors, a fixed chemical featurisation

Everything downstream of the drug-side table is identical across arms: the same
SynergyModel, cell encoder, fragment pool, head, symmetry penalty and grouped
split. Drug subsets are nested within a seed (the size-12 set is a subset of
the size-21 set), which removes sampling noise from the comparison across
sizes, and total rows are capped to a fixed budget so that vocabulary size
varies while data volume does not.

Plot learned-minus-onehot against vocabulary size; where it crosses zero is the
threshold.

Usage
-----
    python experiments/e13_vocab_scaling.py --n-drugs 21 --arm learned \
        --seed 42 --max-rows 20000 --outdir experiments/results/scale_learned_v21_s42
"""

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e12 = load_module(REPO / "experiments/e12_end2end_teacher.py", "e12")
e11 = load_module(REPO / "experiments/e11_build_ecfp_embeddings.py", "e11")
sp = e12.sp


class ScalingModel(nn.Module):
    """One architecture, three sources for the drug-side vectors."""

    def __init__(self, arm, atom_dim, d_graph, d_frag, d_cell, hidden, dropout,
                 gcn_hidden, mol_static=None, frag_static=None):
        super().__init__()
        self.arm = arm
        if arm == "learned":
            self.mol_enc = e12.GraphEncoder(atom_dim, gcn_hidden, d_graph)
            self.frag_enc = e12.GraphEncoder(atom_dim, gcn_hidden, d_frag)
        else:
            self.register_buffer("mol_static", mol_static)
            self.register_buffer("frag_static", frag_static)
        self.core = sp.SynergyModel(d_graph=d_graph, d_frag=d_frag,
                                    d_cell=d_cell, hidden=hidden, dropout=dropout)

    def tables(self, mol_g, frag_g):
        if self.arm == "learned":
            return self.mol_enc(*mol_g), self.frag_enc(*frag_g)
        return self.mol_static, self.frag_static

    def forward(self, cell, ia, ib, fa, fam, fb, fbm, mol_g, frag_g):
        mt, ft = self.tables(mol_g, frag_g)
        return self.core(cell, mt[ia], mt[ib], ft[fa], fam, ft[fb], fbm)


def ecfp_table(smiles_list, n_bits, radius):
    vecs = [e11.ecfp(s, n_bits, radius, True) for s in smiles_list]
    dim = max(len(v) for v in vecs if v is not None)
    M = np.stack([v if v is not None else np.zeros(dim) for v in vecs]).astype(np.float64)
    d = M[:, n_bits:]
    mu, sd = d.mean(0), d.std(0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    M[:, n_bits:] = np.tanh((d - mu) / sd)
    return torch.from_numpy(np.nan_to_num(M, nan=0.0, posinf=1.0,
                                          neginf=-1.0)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO / "clean_runs/almanac_cache"))
    ap.add_argument("--arm", choices=("learned", "onehot", "ecfp"), required=True)
    ap.add_argument("--n-drugs", type=int, required=True)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lambda-sym", type=float, default=0.1)
    ap.add_argument("--d-graph", type=int, default=128)
    ap.add_argument("--d-frag", type=int, default=128)
    ap.add_argument("--gcn-hidden", type=int, default=128)
    ap.add_argument("--d-cell", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--n-bits", type=int, default=1024)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sp.set_seed(a.seed)
    dev = a.device
    C = Path(a.cache)

    meta = pickle.load(open(C / "meta.pkl", "rb"))
    omics = np.load(C / "omics.npy")
    info = json.load(open(C / "vocab.json"))
    vocab = info["vocab"]

    # ---- nested drug subset ---------------------------------------------
    all_drugs = sorted(vocab)
    rng = np.random.default_rng(a.seed)
    order = list(rng.permutation(all_drugs))
    if a.n_drugs > len(order):
        raise SystemExit("only %d encodable drugs available" % len(order))
    chosen = set(order[:a.n_drugs])

    m = meta[meta.drug_a.isin(chosen) & meta.drug_b.isin(chosen)].reset_index(drop=True)
    if len(m) < 500:
        raise SystemExit("only %d rows at n_drugs=%d; too few" % (len(m), a.n_drugs))
    n_avail = len(m)
    if len(m) > a.max_rows:
        sel = rng.choice(len(m), size=a.max_rows, replace=False)
        m = m.iloc[np.sort(sel)].reset_index(drop=True)

    drug_list = sorted(chosen)
    d_idx = {d: i for i, d in enumerate(drug_list)}
    smiles = [vocab[d]["smiles"] for d in drug_list]

    if a.arm == "onehot":
        # identity for the molecule, and one fragment slot carrying the same code
        frag_list = list(drug_list)
        f_idx = {d: i for i, d in enumerate(frag_list)}
        frag_of = {d: [f_idx[d]] for d in drug_list}
        frag_smiles = smiles
    else:
        frag_set = sorted({f for d in drug_list for f in vocab[d]["fragments"]})
        f_idx = {s: i for i, s in enumerate(frag_set)}
        frag_of = {d: [f_idx[f] for f in vocab[d]["fragments"]] for d in drug_list}
        frag_smiles = frag_set

    ia = m.drug_a.map(d_idx).to_numpy()
    ib = m.drug_b.map(d_idx).to_numpy()
    fa = [frag_of[d] for d in m.drug_a]
    fb = [frag_of[d] for d in m.drug_b]
    y = m.y.to_numpy(dtype=np.float32)
    cells = omics[m.cell_idx.to_numpy()]

    print("=" * 70)
    print("  E13  arm=%s  n_drugs=%d  seed=%d" % (a.arm, a.n_drugs, a.seed))
    print("  rows %s of %s available | cells %d | fragments %d"
          % (format(len(m), ","), format(n_avail, ","),
             len(set(m.cell_idx)), len(frag_smiles)))

    # ---- grouped split over (cell line, pair) ----------------------------
    pairs = [tuple(sorted((x, z))) for x, z in zip(m.drug_a, m.drug_b)]
    keys = np.asarray(["%s|%s" % (c, p) for c, p in zip(m.cell_line, pairs)])
    tr, va, te = e12.grouped_split(keys, 0.1, 0.1, a.seed)
    leak = len((set(keys[tr]) & set(keys[te])) | (set(keys[tr]) & set(keys[va])))
    if leak:
        raise SystemExit("grouped split leaked")
    print("  train %d | val %d | test %d | groups %d | leaked 0"
          % (len(tr), len(va), len(te), len(set(keys))))
    print("=" * 70)

    # ---- drug-side tables -------------------------------------------------
    mol_g = frag_g = None
    mol_static = frag_static = None
    atom_dim = 16
    if a.arm == "learned":
        MF, MA, MM, atom_dim, _ = e12.build_graph_tensors(smiles, dev)
        FF, FA_, FM, _, _ = e12.build_graph_tensors(frag_smiles, dev)
        mol_g, frag_g = (MF, MA, MM), (FF, FA_, FM)
        d_graph, d_frag = a.d_graph, a.d_frag
    elif a.arm == "onehot":
        V = len(drug_list)
        mol_static = torch.eye(V)
        frag_static = torch.eye(V)
        d_graph = d_frag = V
    else:
        mol_static = ecfp_table(smiles, a.n_bits, a.radius)
        frag_static = ecfp_table(frag_smiles, a.n_bits, a.radius)
        d_graph, d_frag = mol_static.shape[1], frag_static.shape[1]

    model = ScalingModel(a.arm, atom_dim, d_graph, d_frag, a.d_cell, a.hidden,
                         a.dropout, a.gcn_hidden,
                         mol_static=mol_static, frag_static=frag_static).to(dev)

    if a.arm == "learned":
        enc = list(model.mol_enc.parameters()) + list(model.frag_enc.parameters())
        eids = {id(p) for p in enc}
        rest = [p for p in model.parameters() if id(p) not in eids]
        opt = torch.optim.AdamW([{"params": enc, "lr": a.encoder_lr},
                                 {"params": rest, "lr": a.lr}],
                                weight_decay=a.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                weight_decay=a.weight_decay)

    def subset(idx):
        return e12.IdxDataset(cells[idx], ia[idx], ib[idx],
                              [fa[i] for i in idx], [fb[i] for i in idx], y[idx])

    loaders = {k: DataLoader(subset(v), batch_size=a.batch_size,
                             shuffle=(k == "train"), collate_fn=e12.collate)
               for k, v in (("train", tr), ("val", va), ("test", te))}

    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for cell, i_a, i_b, f_a, m_a, f_b, m_b, yy in loaders["train"]:
            cell, yy = cell.to(dev), yy.to(dev)
            i_a, i_b = i_a.to(dev), i_b.to(dev)
            f_a, m_a, f_b, m_b = f_a.to(dev), m_a.to(dev), f_b.to(dev), m_b.to(dev)
            p_ab = model(cell, i_a, i_b, f_a, m_a, f_b, m_b, mol_g, frag_g)
            p_ba = model(cell, i_b, i_a, f_b, m_b, f_a, m_a, mol_g, frag_g)
            loss = F.mse_loss(p_ab, yy) + a.lambda_sym * F.mse_loss(p_ab, p_ba)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            tot += float(loss.item()) * yy.shape[0]
            n += yy.shape[0]

        vm = e12.run_eval(model, loaders["val"], mol_g, frag_g, dev)
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1),
                         val_rmse=vm["rmse"], val_pearson=vm["pearson"]))
        if ep % 10 == 0 or ep == 1:
            print("Epoch %03d | loss=%.3f | val RMSE=%.4f | val PCC=%.4f"
                  % (ep, tot / max(n, 1), vm["rmse"], vm["pearson"]), flush=True)
        if vm["rmse"] < best - 1e-5:
            best, bad = vm["rmse"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop at epoch %d" % ep)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    res = {"metrics_%s" % k: e12.run_eval(model, loaders[k], mol_g, frag_g, dev)
           for k in ("train", "val", "test")}
    for k, v in res.items():
        print("%-14s RMSE=%.4f Pearson=%.4f" % (k, v["rmse"], v["pearson"]))

    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    json.dump(dict(arm=a.arm, n_drugs=a.n_drugs, seed=a.seed,
                   rows_used=int(len(m)), rows_available=int(n_avail),
                   max_rows=a.max_rows, n_fragments=len(frag_smiles),
                   n_cells=int(len(set(m.cell_idx))), n_groups=int(len(set(keys))),
                   d_graph=int(d_graph), d_frag=int(d_frag),
                   epochs_run=len(hist), best_val_rmse=float(best),
                   drugs=drug_list),
              open(out / "run_audit.json", "w"), indent=1)
    import pandas as pd
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    print("\nwritten -> %s" % out)


if __name__ == "__main__":
    main()
