"""
E12 -- end-to-end teacher: train the graph encoders on the synergy objective
============================================================================
Diagnosis (E8/E10): the molecule and fragment embeddings are produced by graph
autoencoders trained to *reconstruct* molecules and then frozen. Reconstruction
preserves atoms and bonds but has no reason to preserve what determines
synergy, and the result is that a 21-way one-hot drug code outperforms the
learned embeddings under grouped splits, while the same substitution applied to
DeepSynergy hurts it.

The fix tested here: delete the autoencoder pretraining stage entirely,
initialise both GCN encoders randomly, and let gradients from the synergy loss
flow into the message-passing layers.

Everything else is held fixed and reused directly from synergy_prediction.py --
the same cell encoder, the same fragment attention pool, the same fusion head,
the same symmetry penalty, the same Loewe target and the same row set. Only the
source of the drug-side vectors changes.

Because the encoders now do the hard work they get their own, lower learning
rate; the head keeps the original one.

The decision rule is the one-hot control, not the raw number: this run counts as
progress only if it beats one-hot drug identity on the same splits and seeds
(0.853 cellpair, 0.728 pair).

Usage
-----
    python experiments/e12_end2end_teacher.py --group-mode cellpair --seed 42 \
        --outdir experiments/results/e2e_cellpair_s42

    # after a successful run, emit embeddings the existing pipeline can consume
    python experiments/e12_end2end_teacher.py ... --export-embeddings DIR
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
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
RUN0 = REPO / "clean_runs/2026-04-29_full_rerun"

DEFAULTS = dict(
    input="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl",
    molecule_embeddings=str(RUN0 / "molecule_ae/molecule_smiles_embeddings.pkl"),
    fragment_embeddings=str(RUN0 / "fragment_ae/fragment_smiles_embeddings.pkl"),
    fragment_map=str(RUN0 / "fragment_ae/smiles_to_fragments.pkl"),
    drug_smiles_map=str(REPO / "everything_drug_smiles_map.csv"),
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sp = load_module(REPO / "synergy_prediction.py", "sp_teacher")
ae = load_module(REPO / "drug_graph_autoencoder.py", "ae_graph")


def norm(x):
    return str(x).strip().lower()


# ------------------------------------------------------------------ splits --
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


# ------------------------------------------------------------------ graphs --
def build_graph_tensors(smiles_list, device):
    """Pad every molecule in a fixed vocabulary into one batched tensor."""
    feats, adjs, nfail = [], [], 0
    for s in smiles_list:
        try:
            g = ae.smiles_to_graph(s, s)
            feats.append(g.node_features)
            adjs.append(g.adjacency)
        except Exception:
            nfail += 1
            feats.append(torch.zeros(1, 16, dtype=torch.float32))
            adjs.append(torch.ones(1, 1, dtype=torch.float32))
    max_n = max(f.shape[0] for f in feats)
    dim = feats[0].shape[1]
    V = len(feats)
    F_ = torch.zeros(V, max_n, dim)
    A_ = torch.zeros(V, max_n, max_n)
    M_ = torch.zeros(V, max_n)
    for i, (f, a) in enumerate(zip(feats, adjs)):
        k = f.shape[0]
        F_[i, :k] = f
        A_[i, :k, :k] = a
        M_[i, :k] = 1.0
    return F_.to(device), A_.to(device), M_.to(device), dim, nfail


class GraphEncoder(nn.Module):
    """Same two-layer GCN as the autoencoder's encoder, trained from scratch."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gcn1 = ae.GCNLayer(in_dim, hidden_dim)
        self.gcn2 = ae.GCNLayer(hidden_dim, out_dim)

    def forward(self, features, adjacency, mask):
        h = F.relu(self.gcn1(features, adjacency))
        z = self.gcn2(h, adjacency)
        z = z * mask.unsqueeze(-1)
        return z.sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)


class End2EndSynergy(nn.Module):
    """Trainable encoders in front of the unmodified SynergyModel."""

    def __init__(self, atom_dim, d_graph, d_frag, d_cell, hidden, dropout,
                 gcn_hidden):
        super().__init__()
        self.mol_enc = GraphEncoder(atom_dim, gcn_hidden, d_graph)
        self.frag_enc = GraphEncoder(atom_dim, gcn_hidden, d_frag)
        self.core = sp.SynergyModel(d_graph=d_graph, d_frag=d_frag,
                                    d_cell=d_cell, hidden=hidden,
                                    dropout=dropout)

    def tables(self, mol_g, frag_g):
        return self.mol_enc(*mol_g), self.frag_enc(*frag_g)

    def forward(self, cell, ia, ib, fa, fa_mask, fb, fb_mask, mol_g, frag_g):
        mol_tab, frag_tab = self.tables(mol_g, frag_g)
        return self.core(cell, mol_tab[ia], mol_tab[ib],
                         frag_tab[fa], fa_mask, frag_tab[fb], fb_mask)


# ----------------------------------------------------------------- dataset --
class IdxDataset(Dataset):
    def __init__(self, cells, ia, ib, fa, fb, y):
        self.cells, self.ia, self.ib = cells, ia, ib
        self.fa, self.fb, self.y = fa, fb, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.cells[i], self.ia[i], self.ib[i], self.fa[i], self.fb[i], self.y[i]


def collate(batch):
    cells, ia, ib, fa, fb, y = zip(*batch)
    n = len(batch)
    ma = max(len(f) for f in fa)
    mb = max(len(f) for f in fb)
    FA = torch.zeros(n, ma, dtype=torch.long)
    FB = torch.zeros(n, mb, dtype=torch.long)
    MA = torch.zeros(n, ma, dtype=torch.bool)
    MB = torch.zeros(n, mb, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(fa, fb)):
        FA[i, :len(a)] = torch.as_tensor(a, dtype=torch.long)
        MA[i, :len(a)] = True
        FB[i, :len(b)] = torch.as_tensor(b, dtype=torch.long)
        MB[i, :len(b)] = True
    return (torch.from_numpy(np.stack(cells).astype(np.float32)),
            torch.as_tensor(ia, dtype=torch.long),
            torch.as_tensor(ib, dtype=torch.long),
            FA, MA, FB, MB,
            torch.as_tensor(np.asarray(y, dtype=np.float32)))


def metrics(y, p):
    return sp._metric_dict(np.asarray(y), np.asarray(p))


def run_eval(model, loader, mol_g, frag_g, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for cell, ia, ib, fa, fam, fb, fbm, y in loader:
            out = model(cell.to(device), ia.to(device), ib.to(device),
                        fa.to(device), fam.to(device), fb.to(device),
                        fbm.to(device), mol_g, frag_g)
            ys.append(y.numpy())
            ps.append(out.detach().cpu().numpy())
    return metrics(np.concatenate(ys), np.concatenate(ps))


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument("--group-mode", choices=("cellpair", "pair", "random"),
                    default="cellpair")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42,
                    help="controls the data partition")
    ap.add_argument("--init-seed", type=int, default=None,
                    help="controls weight initialisation and batch order; "
                         "defaults to --seed. Set separately to test whether a "
                         "failed run is a property of the partition or of the "
                         "initialisation.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3, help="head / cell encoder")
    ap.add_argument("--encoder-lr", type=float, default=1e-4, help="GCN encoders")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lambda-sym", type=float, default=0.1)
    ap.add_argument("--d-graph", type=int, default=128)
    ap.add_argument("--d-frag", type=int, default=128)
    ap.add_argument("--gcn-hidden", type=int, default=128)
    ap.add_argument("--d-cell", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--export-embeddings", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sp.set_seed(a.seed)
    device = a.device

    # ---- exactly the teacher's row set ----------------------------------
    ex = sp._build_examples(
        dataset=sp.load_pickle(Path(a.input)),
        molecule_embeddings=sp.load_embedding_pickle(Path(a.molecule_embeddings)),
        fragment_embeddings=sp.load_embedding_pickle(Path(a.fragment_embeddings)),
        fragment_map=sp.load_pickle(Path(a.fragment_map)),
        drug_smiles_map=sp.load_drug_smiles_map(Path(a.drug_smiles_map)),
        include_no_fragments=False,
    )
    meta = ex["meta"].reset_index(drop=True)
    y = ex["targets"].astype(np.float32)
    cells = ex["cells"]
    frag_map = sp.load_pickle(Path(a.fragment_map))
    print("rows: %s (matches teacher)" % format(len(meta), ","))

    # ---- vocabularies ----------------------------------------------------
    sa = meta.drug_a_smiles.astype(str).str.strip().tolist()
    sb = meta.drug_b_smiles.astype(str).str.strip().tolist()
    drug_vocab = sorted(set(sa) | set(sb))
    d_idx = {s: i for i, s in enumerate(drug_vocab)}

    fmap = {}
    for s in drug_vocab:
        fmap[s] = [str(x).strip() for x in (frag_map.get(s) or [])]
    frag_vocab = sorted({f for v in fmap.values() for f in v})
    f_idx = {s: i for i, s in enumerate(frag_vocab)}
    frag_of = {s: [f_idx[f] for f in fmap[s]] or [0] for s in drug_vocab}

    print("vocabulary: %d drugs, %d fragments" % (len(drug_vocab), len(frag_vocab)))

    MF, MA, MM, atom_dim, nf1 = build_graph_tensors(drug_vocab, device)
    FF, FA_, FM, _, nf2 = build_graph_tensors(frag_vocab, device)
    mol_g, frag_g = (MF, MA, MM), (FF, FA_, FM)
    if nf1 or nf2:
        print("  unparseable graphs: %d drugs, %d fragments" % (nf1, nf2))

    ia = np.asarray([d_idx[s] for s in sa])
    ib = np.asarray([d_idx[s] for s in sb])
    fa = [frag_of[s] for s in sa]
    fb = [frag_of[s] for s in sb]

    # ---- split -----------------------------------------------------------
    if a.group_mode == "random":
        rng = np.random.default_rng(a.seed)
        perm = rng.permutation(len(meta))
        nt = int(round(len(meta) * a.test_fraction))
        nv = int(round(len(meta) * a.val_fraction))
        te, va, tr = perm[:nt], perm[nt:nt + nv], perm[nt + nv:]
        leak, ngroups = -1, -1
    else:
        keys = group_keys(meta, a.group_mode)
        tr, va, te = grouped_split(keys, a.val_fraction, a.test_fraction, a.seed)
        ktr, kva, kte = set(keys[tr]), set(keys[va]), set(keys[te])
        leak = len((ktr & kte) | (ktr & kva) | (kva & kte))
        ngroups = len(set(keys))
        if leak > 0:
            raise SystemExit("grouped split leaked; aborting")

    def subset(idx):
        return IdxDataset(cells[idx], ia[idx], ib[idx],
                          [fa[i] for i in idx], [fb[i] for i in idx], y[idx])

    loaders = {k: DataLoader(subset(v), batch_size=a.batch_size,
                             shuffle=(k == "train"), collate_fn=collate)
               for k, v in (("train", tr), ("val", va), ("test", te))}

    print("=" * 68)
    print("  E12 end-to-end | split=%s | seed=%d" % (a.group_mode, a.seed))
    print("  train %d | val %d | test %d | groups %s | leaked %s"
          % (len(tr), len(va), len(te), ngroups, leak))
    print("=" * 68)

    # ---- model -----------------------------------------------------------
    # Re-seed here so that initialisation can be varied independently of the
    # partition, which --seed fixed above.
    init_seed = a.seed if a.init_seed is None else a.init_seed
    sp.set_seed(init_seed)
    model = End2EndSynergy(atom_dim, a.d_graph, a.d_frag, a.d_cell, a.hidden,
                           a.dropout, a.gcn_hidden).to(device)
    enc_params = list(model.mol_enc.parameters()) + list(model.frag_enc.parameters())
    enc_ids = {id(p) for p in enc_params}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": a.encoder_lr},
         {"params": rest, "lr": a.lr}], weight_decay=a.weight_decay)
    n_enc = sum(p.numel() for p in enc_params)
    print("  parameters: %s encoder + %s rest"
          % (format(n_enc, ","), format(sum(p.numel() for p in rest), ",")))

    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for cell, i_a, i_b, f_a, m_a, f_b, m_b, yy in loaders["train"]:
            cell, yy = cell.to(device), yy.to(device)
            i_a, i_b = i_a.to(device), i_b.to(device)
            f_a, m_a = f_a.to(device), m_a.to(device)
            f_b, m_b = f_b.to(device), m_b.to(device)

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

        vm = run_eval(model, loaders["val"], mol_g, frag_g, device)
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1), **{"val_" + k: v for k, v in vm.items()}))
        print("Epoch %03d | train loss=%.4f | val RMSE=%.4f | val Pearson=%.4f"
              % (ep, tot / max(n, 1), vm["rmse"], vm["pearson"]))

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

    res = {"metrics_%s" % k: run_eval(model, loaders[k], mol_g, frag_g, device)
           for k in ("train", "val", "test")}
    for k, v in res.items():
        print("%-14s RMSE=%.4f Pearson=%.4f Spearman=%.4f"
              % (k, v["rmse"], v["pearson"], v["spearman"]))

    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    json.dump(dict(split=a.group_mode, seed=a.seed, init_seed=int(init_seed),
                   leaked_groups=int(leak),
                   n_groups=int(ngroups), n_rows=int(len(meta)),
                   n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
                   atom_dim=int(atom_dim), d_graph=a.d_graph, d_frag=a.d_frag,
                   gcn_hidden=a.gcn_hidden, encoder_lr=a.encoder_lr, lr=a.lr,
                   encoder_params=int(n_enc), epochs_run=len(hist),
                   best_val_rmse=float(best),
                   model="end-to-end GCN encoders + unmodified SynergyModel"),
              open(out / "run_audit.json", "w"), indent=1)
    import pandas as pd
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "molecule_dim": a.d_graph,
                "fragment_dim": a.d_frag, "d_cell": a.d_cell,
                "hidden": a.hidden, "dropout": a.dropout, "seed": a.seed,
                "input_dataset": a.input, "target": "loewe",
                "include_no_fragments": False}, out / "model.pt")

    # ---- optional: emit embeddings the existing pipeline can consume -----
    if a.export_embeddings:
        ed = Path(a.export_embeddings)
        ed.mkdir(parents=True, exist_ok=True)
        model.eval()
        with torch.no_grad():
            mt, ft = model.tables(mol_g, frag_g)
        mt, ft = mt.cpu().numpy(), ft.cpu().numpy()
        pickle.dump({s: mt[i].astype(np.float32) for s, i in d_idx.items()},
                    open(ed / "molecule_smiles_embeddings.pkl", "wb"))
        pickle.dump({s: ft[i].astype(np.float32) for s, i in f_idx.items()},
                    open(ed / "fragment_smiles_embeddings.pkl", "wb"))
        print("exported learned embeddings -> %s" % ed)

    print("\nwritten -> %s" % out)
    print("DECISION RULE: beats one-hot only if test Pearson > "
          "%.3f" % (0.853 if a.group_mode == "cellpair" else 0.728))


if __name__ == "__main__":
    main()
