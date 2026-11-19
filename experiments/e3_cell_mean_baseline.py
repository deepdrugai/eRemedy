"""
E3 -- Cell-mean baseline: how much are the chemistry channels adding?
=====================================================================
Modality SHAP gives cell context 38.01 against 11.38 for fragments and 11.16
for whole-molecule structure. The obvious control is therefore a predictor
that ignores chemistry entirely.

Three trivial baselines, evaluated on the teacher's own 80/10/10 split with a
matched random seed:

  B0  global mean          -- predict the dataset mean for everything
  B1  cell mean            -- predict each cell line's mean synergy
  B2  drug-pair mean       -- predict each unordered drug pair's mean
  B3  cell x pair mean     -- the full lookup table, backing off when unseen

B3 is the ceiling for a model that memorises rather than reasons. If it
approaches the teacher's 0.946, the architecture is doing lookup.

No training, no GPU. Read-only on all existing artefacts.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

OUT = "experiments/results"


def norm(x):
    return str(x).strip().lower()


def metrics(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    ok = np.isfinite(y) & np.isfinite(yh)
    y, yh = y[ok], yh[ok]
    rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
    mae = float(np.mean(np.abs(y - yh)))
    pcc = float(np.corrcoef(y, yh)[0, 1]) if np.std(yh) > 1e-9 else 0.0
    ss = float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(n=len(y), rmse=rmse, mae=mae, pcc=pcc, r2=ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl")
    ap.add_argument("--cache", default="experiments/results/_oneil_meta.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restrict-vocab", action="store_true",
                    help="keep only the teacher's 21 encodable drugs, so the "
                         "comparison against its reported metrics is exact")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    VOCAB = {"391210-10-9", "7-ethyl-10-hydroxycamptothecin", "915019-65-7",
             "bortezomib", "dasatinib", "dinaciclib", "mk-1775", "mk-2206",
             "mk-4541", "mrk003", "veliparib", "cyclophosphamide", "etoposide",
             "geldanamycin", "gemcitabine", "metformin", "methotrexate",
             "mitomycin c", "temozolomide", "vinblastine", "vinorelbine"}

    if os.path.exists(a.cache):
        m = pd.read_csv(a.cache)
    else:
        ds = pickle.load(open(a.dataset, "rb"))
        m = ds["meta"][["cell_line_name", "drug_a", "drug_b", "y_loewe"]].copy()
        m.to_csv(a.cache, index=False)
    m = m.dropna(subset=["y_loewe"]).reset_index(drop=True)
    m["cell"] = m.cell_line_name.map(norm)
    m["pair"] = [tuple(sorted((norm(x), norm(y))))
                 for x, y in zip(m.drug_a, m.drug_b)]
    tag = "all drugs"
    if a.restrict_vocab:
        keep = [p[0] in VOCAB and p[1] in VOCAB for p in m.pair]
        m = m[keep].reset_index(drop=True)
        tag = "teacher's 21-drug vocabulary only"

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(m))
    n_tr = int(0.8 * len(m))
    n_va = int(0.1 * len(m))
    tr = m.iloc[idx[:n_tr]]
    te = m.iloc[idx[n_tr + n_va:]]

    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 78)
    say("  E3  CELL-MEAN AND LOOKUP BASELINES")
    say("=" * 78)
    say("")
    say("rows %s | train %s | test %s | cell lines %d | drug pairs %d"
        % (format(len(m), ","), format(len(tr), ","), format(len(te), ","),
           m.cell.nunique(), m.pair.nunique()))
    say("seed %d, 80/10/10 split matching the teacher's protocol" % a.seed)
    say("scope: %s" % tag)
    say("")

    gm = tr.y_loewe.mean()
    cm = tr.groupby("cell").y_loewe.mean()
    pm = tr.groupby("pair").y_loewe.mean()
    cpm = tr.groupby(["cell", "pair"]).y_loewe.mean()

    pred = {
        "B0  global mean": np.full(len(te), gm),
        "B1  cell mean": te.cell.map(cm).fillna(gm).values,
        "B2  drug-pair mean": te.pair.map(pm).fillna(gm).values,
    }
    b3 = [cpm.get((c, p), np.nan) for c, p in zip(te.cell, te.pair)]
    b3 = pd.Series(b3)
    fb = te.cell.map(cm).reset_index(drop=True).fillna(gm)
    cov = float(b3.notna().mean())
    pred["B3  cell x pair lookup"] = b3.fillna(fb).values

    say("-" * 78)
    say("Test-set performance")
    say("-" * 78)
    say("  %-26s %8s %8s %8s %8s" % ("baseline", "RMSE", "MAE", "PCC", "R2"))
    say("  " + "-" * 62)
    rows = []
    for k, v in pred.items():
        s = metrics(te.y_loewe.values, v)
        say("  %-26s %8.3f %8.3f %8.4f %8.4f"
            % (k, s["rmse"], s["mae"], s["pcc"], s["r2"]))
        rows.append(dict(baseline=k, **s))
    say("  " + "-" * 62)
    say("  %-26s %8.3f %8.3f %8.4f %8.4f"
        % ("TEACHER (reported)", 6.181, 4.453, 0.9463, 0.8940))
    say("")
    say("  B3 lookup coverage of test rows: %.1f%% (rest backed off to cell mean)"
        % (100 * cov))
    say("")
    sfx = "_vocab" if a.restrict_vocab else ""
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e3_baselines%s.csv" % sfx),
                              index=False)

    say("-" * 78)
    say("Where does the variance live?")
    say("-" * 78)
    tot = m.y_loewe.var()
    bc = m.groupby("cell").y_loewe.mean().var()
    bp = m.groupby("pair").y_loewe.mean().var()
    say("  total variance in Loewe score      : %8.2f" % tot)
    say("  variance of cell-line means        : %8.2f  (%.1f%% of total)"
        % (bc, 100 * bc / tot))
    say("  variance of drug-pair means        : %8.2f  (%.1f%% of total)"
        % (bp, 100 * bp / tot))
    say("")

    say("=" * 78)
    say("  VERDICT")
    say("=" * 78)
    say("")
    p1 = [r for r in rows if r["baseline"].startswith("B1")][0]["pcc"]
    p3 = [r for r in rows if r["baseline"].startswith("B3")][0]["pcc"]
    say("  cell mean alone        : PCC %.4f" % p1)
    say("  cell x pair lookup     : PCC %.4f" % p3)
    say("  teacher                : PCC 0.9463")
    say("")
    say("  headroom above cell mean captured by the teacher : %.1f%%"
        % (100 * (0.9463 - p1) / (1 - p1)))
    if p3 > 0.90:
        say("")
        say("  A pure lookup table reaches PCC %.3f. The teacher's 0.946 is" % p3)
        say("  therefore largely attainable without any molecular representation,")
        say("  and Contribution 1 must be reframed accordingly. Run the one-hot")
        say("  controls (E2 in the review) to confirm on the trained model.")
    elif p1 > 0.60:
        say("")
        say("  Cell context alone reaches PCC %.3f, so a substantial share of" % p1)
        say("  the headline correlation is context rather than chemistry.")
    else:
        say("")
        say("  Cell context alone reaches only PCC %.3f, so the chemistry" % p1)
        say("  channels are carrying real signal on this split.")
    say("")

    with open(os.path.join(OUT, "e3_cell_mean_baseline%s.txt" % sfx), "w") as f:
        f.write("\n".join(L) + "\n")
    print("written -> %s/e3_cell_mean_baseline%s.txt" % (OUT, sfx))


if __name__ == "__main__":
    main()
