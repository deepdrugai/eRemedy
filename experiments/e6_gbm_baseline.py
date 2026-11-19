"""
E6 -- Gradient boosting baseline under both splits
==================================================
How much does the multi-modal deep architecture add over gradient boosting on
the same molecular features? Run under both evaluation protocols so the answer
is not confounded by the leakage documented in E3.

Features per row, symmetric in the two drugs by construction:
    [ e_A + e_B , |e_A - e_B| , one-hot(cell line) ]
using the same 128-d molecule embeddings the teacher consumes, so the
comparison isolates the architecture rather than the representation.

Splits:
    random    the teacher's original protocol (leaky: ~4 rows per combination)
    cellpair  whole (cell line, drug pair) groups held out
    pair      whole drug pairs held out across all cell lines

Read-only on existing artefacts. Writes only into experiments/results/.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = Path(__file__).resolve().parents[1]
RUN0 = REPO / "clean_runs/2026-04-29_full_rerun"
OUT = REPO / "experiments/results"

TEACHER_REPORTED = {"rmse": 6.181, "mae": 4.453, "pcc": 0.9463, "r2": 0.8940}


def norm(x):
    return str(x).strip().lower()


def metrics(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return dict(n=int(len(y)),
                rmse=float(np.sqrt(np.mean((y - yh) ** 2))),
                mae=float(np.mean(np.abs(y - yh))),
                pcc=float(np.corrcoef(y, yh)[0, 1]),
                r2=float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)))


def build(dataset_path, mol_emb_path):
    ds = pickle.load(open(dataset_path, "rb"))
    meta = ds["meta"]
    emb = pickle.load(open(mol_emb_path, "rb"))
    emb = {norm(k): np.asarray(v, dtype=np.float32) for k, v in emb.items()}

    smap = pd.read_csv(REPO / "everything_drug_smiles_map.csv")
    cols = {c.lower(): c for c in smap.columns}
    ncol = cols.get("drug_name") or cols.get("name") or smap.columns[0]
    scol = cols.get("smiles") or smap.columns[1]
    name2smi = {norm(n): str(s) for n, s in zip(smap[ncol], smap[scol])}

    def vec(name):
        n = norm(name)
        if n in emb:
            return emb[n]
        s = name2smi.get(n)
        return emb.get(norm(s)) if s else None

    X, y, keys_cp, keys_p, cells = [], [], [], [], []
    for r in meta.itertuples(index=False):
        if not np.isfinite(getattr(r, "y_loewe", np.nan)):
            continue
        ea, eb = vec(r.drug_a), vec(r.drug_b)
        if ea is None or eb is None:
            continue
        X.append(np.concatenate([ea + eb, np.abs(ea - eb)]))
        y.append(float(r.y_loewe))
        p = str(tuple(sorted((norm(r.drug_a), norm(r.drug_b)))))
        c = norm(r.cell_line_name)
        keys_cp.append("%s|%s" % (c, p))
        keys_p.append(p)
        cells.append(c)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    cu = sorted(set(cells))
    ci = {c: i for i, c in enumerate(cu)}
    oh = np.zeros((len(cells), len(cu)), dtype=np.float32)
    oh[np.arange(len(cells)), [ci[c] for c in cells]] = 1.0
    return (np.hstack([X, oh]), y, np.asarray(keys_cp),
            np.asarray(keys_p), np.asarray(cells))


def split(mode, keys_cp, keys_p, n, seed, val=0.1, test=0.1):
    rng = np.random.default_rng(seed)
    if mode == "random":
        idx = rng.permutation(n)
        nt, nv = int(round(n * test)), int(round(n * val))
        return idx[nt + nv:], idx[nt:nt + nv], idx[:nt]
    keys = keys_cp if mode == "cellpair" else keys_p
    uniq = np.unique(keys)
    rng.shuffle(uniq)
    g = len(uniq)
    nt, nv = max(1, int(round(g * test))), max(1, int(round(g * val)))
    te_g, va_g = set(uniq[:nt]), set(uniq[nt:nt + nv])
    idx = np.arange(n)
    te = idx[[k in te_g for k in keys]]
    va = idx[[k in va_g for k in keys]]
    tr = idx[[(k not in te_g and k not in va_g) for k in keys]]
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl")
    ap.add_argument("--mol-emb",
                    default=str(RUN0 / "molecule_ae/molecule_smiles_embeddings.pkl"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    X, y, kcp, kp, cells = build(a.dataset, a.mol_emb)
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 78)
    say("  E6  GRADIENT BOOSTING BASELINE UNDER BOTH SPLITS")
    say("=" * 78)
    say("")
    say("rows %s | features %d | cell lines %d | (cell,pair) groups %s | pairs %d"
        % (format(len(y), ","), X.shape[1], len(set(cells)),
           format(len(set(kcp)), ","), len(set(kp))))
    say("features: [e_A + e_B | |e_A - e_B| | one-hot(cell)], symmetric by construction")
    say("")

    rows = []
    for mode in ("random", "cellpair", "pair"):
        tr, va, te = split(mode, kcp, kp, len(y), a.seed)
        model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_depth=None,
            early_stopping=True, validation_fraction=None,
            random_state=a.seed)
        model.fit(X[tr], y[tr])
        s = metrics(y[te], model.predict(X[te]))
        s.update(split_mode=mode, n_train=len(tr), n_test=len(te))
        rows.append(s)
        say("  %-9s  train %6d  test %5d   RMSE %6.3f  MAE %6.3f  PCC %+.4f  R2 %+.4f"
            % (mode, len(tr), len(te), s["rmse"], s["mae"], s["pcc"], s["r2"]))

    say("")
    say("-" * 78)
    say("Comparison")
    say("-" * 78)
    R = {r["split_mode"]: r for r in rows}
    say("  %-34s %8s %8s" % ("", "PCC", "RMSE"))
    say("  " + "-" * 52)
    say("  %-34s %8.4f %8.3f" % ("GBM, random split (leaky)",
                                 R["random"]["pcc"], R["random"]["rmse"]))
    say("  %-34s %8.4f %8.3f" % ("GBM, grouped by (cell,pair)",
                                 R["cellpair"]["pcc"], R["cellpair"]["rmse"]))
    say("  %-34s %8.4f %8.3f" % ("GBM, grouped by pair",
                                 R["pair"]["pcc"], R["pair"]["rmse"]))
    say("  %-34s %8.4f %8.3f" % ("TEACHER, random split (reported)",
                                 TEACHER_REPORTED["pcc"], TEACHER_REPORTED["rmse"]))
    say("")
    say("  GBM drop, random -> (cell,pair) grouped : %+.4f PCC"
        % (R["cellpair"]["pcc"] - R["random"]["pcc"]))
    say("  GBM drop, random -> pair grouped        : %+.4f PCC"
        % (R["pair"]["pcc"] - R["random"]["pcc"]))
    say("")

    say("=" * 78)
    say("  VERDICT")
    say("=" * 78)
    say("")
    if R["random"]["pcc"] >= TEACHER_REPORTED["pcc"] - 0.02:
        say("  Gradient boosting on the same embeddings reaches PCC %.4f under"
            % R["random"]["pcc"])
        say("  the teacher's own protocol, against the teacher's %.4f. The deep"
            % TEACHER_REPORTED["pcc"])
        say("  architecture is not the source of the headline number.")
    else:
        say("  The teacher (%.4f) exceeds gradient boosting (%.4f) on the random"
            % (TEACHER_REPORTED["pcc"], R["random"]["pcc"]))
        say("  split, so the architecture contributes beyond the features.")
    say("")
    say("  Under a leakage-free grouped split the GBM falls to PCC %.4f."
        % R["cellpair"]["pcc"])
    say("  The teacher's grouped-split number is being computed separately;")
    say("  compare the two once it lands.")
    say("")

    pd.DataFrame(rows).to_csv(OUT / "e6_gbm_baseline.csv", index=False)
    json.dump(rows, open(OUT / "e6_gbm_baseline.json", "w"), indent=1)
    with open(OUT / "e6_gbm_baseline.txt", "w") as f:
        f.write("\n".join(L) + "\n")
    print("written -> %s/e6_gbm_baseline.txt" % OUT)


if __name__ == "__main__":
    main()
