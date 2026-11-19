"""
diagnose_lodo.py
================
Work out WHY leave-one-drug-out performance varies so much across the 21
O'Neil drugs, so that Flow B can be gated on something defensible.

Three hypotheses are tested:

  H1  Structural coverage -- PCC tracks Tanimoto similarity to the drugs that
      remain in training.
  H2  Mechanistic coverage -- PCC tracks whether another training drug shares
      the held-out drug's mechanism of action / target.
  H3  Label variance artefact -- low PCC simply reflects a held-out drug whose
      true synergies barely vary, which makes correlation an unstable metric.
      If so, a normalised error is the fairer measure.
"""

import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

LODO = "clean_runs/lodo/lodo_per_drug_ALL.csv"
DRUG_MAP = "everything_drug_smiles_map.csv"
MOA = "drug_moa_cache.csv"
DATA = "oneil_synergy_dataset.csv"
OUT = "thesis/aggregate"
os.makedirs(OUT, exist_ok=True)


def norm(s):
    return str(s).strip().lower()


def main():
    lodo = pd.read_csv(LODO)
    drugs = lodo.drug.tolist()

    # ---------------- H1: structural coverage ----------------
    dm = pd.read_csv(DRUG_MAP)
    name2smi = {norm(r.drug): r.smiles for r in dm.itertuples()}
    fps = {}
    for d in drugs:
        smi = name2smi.get(norm(d))
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                fps[d] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

    struct = []
    for d in drugs:
        if d not in fps:
            struct.append({"drug": d, "max_tanimoto": np.nan, "n_sim_above_0p3": 0})
            continue
        sims = [DataStructs.TanimotoSimilarity(fps[d], fps[o]) for o in fps if o != d]
        struct.append({
            "drug": d,
            "max_tanimoto": max(sims),
            "n_sim_above_0p3": int(sum(1 for s in sims if s >= 0.3)),
        })
    struct = pd.DataFrame(struct)

    # ---------------- H2: mechanistic coverage ----------------
    moa_cov = []
    if os.path.exists(MOA):
        moa = pd.read_csv(MOA)
        moa["key"] = moa.iloc[:, 0].map(norm)
        # collect MoA / target text per drug
        text_cols = [c for c in moa.columns
                     if any(k in c.lower() for k in
                            ("mechanism", "action_type", "target", "atc"))]
        bag = {}
        for _, r in moa.iterrows():
            toks = set()
            for c in text_cols:
                v = r.get(c)
                if isinstance(v, str) and v.strip():
                    for piece in v.split(","):
                        p = piece.strip().lower()
                        if p and p not in ("nan", "unknown", "none"):
                            toks.add(p)
            if toks:
                bag.setdefault(r["key"], set()).update(toks)

        for d in drugs:
            mine = bag.get(norm(d), set())
            if not mine:
                moa_cov.append({"drug": d, "moa_covered": np.nan,
                                "moa_partner": None})
                continue
            best, bestn = None, 0
            for o in drugs:
                if o == d:
                    continue
                shared = mine & bag.get(norm(o), set())
                if len(shared) > bestn:
                    best, bestn = o, len(shared)
            moa_cov.append({"drug": d, "moa_covered": int(bestn > 0),
                            "moa_partner": best})
    moa_cov = pd.DataFrame(moa_cov) if moa_cov else pd.DataFrame(
        {"drug": drugs, "moa_covered": np.nan, "moa_partner": None})

    # ---------------- H3: label variance ----------------
    var = []
    if os.path.exists(DATA):
        raw = pd.read_csv(DATA)
        cols = {c.lower(): c for c in raw.columns}
        ca = cols.get("drug_a") or cols.get("druga")
        cb = cols.get("drug_b") or cols.get("drugb")
        cy = (cols.get("synergy_loewe") or cols.get("loewe")
              or cols.get("synergy") or cols.get("target"))
        if ca and cb and cy:
            raw["_a"] = raw[ca].map(norm)
            raw["_b"] = raw[cb].map(norm)
            for d in drugs:
                k = norm(d)
                sub = raw[(raw._a == k) | (raw._b == k)]
                var.append({
                    "drug": d,
                    "n_rows": len(sub),
                    "label_std": float(sub[cy].std()) if len(sub) > 1 else np.nan,
                    "label_mean": float(sub[cy].mean()) if len(sub) else np.nan,
                })
    var = pd.DataFrame(var) if var else pd.DataFrame({"drug": drugs})

    df = lodo.merge(struct, on="drug", how="left") \
             .merge(moa_cov, on="drug", how="left") \
             .merge(var, on="drug", how="left")

    if "label_std" in df and df.label_std.notna().any():
        df["norm_rmse"] = df.rmse / df.label_std

    df = df.sort_values("pearson", ascending=False)

    show = ["drug", "pearson", "rmse", "max_tanimoto", "n_sim_above_0p3",
            "moa_covered", "label_std", "norm_rmse"]
    show = [c for c in show if c in df.columns]
    print("=" * 100)
    print("  LODO DIAGNOSTICS")
    print("=" * 100)
    print(df[show].to_string(index=False, float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 100)
    print("  HYPOTHESIS TESTS  (correlation with LODO Pearson)")
    print("=" * 100)
    for col, label in [("max_tanimoto", "H1  max Tanimoto to training drugs"),
                       ("n_sim_above_0p3", "H1b count of analogs (T>=0.3)"),
                       ("moa_covered", "H2  shares MoA with a training drug"),
                       ("label_std", "H3  std of held-out drug's labels"),
                       ("n_rows", "    n training rows for that drug")]:
        if col not in df.columns:
            continue
        s = df[[col, "pearson"]].dropna()
        if len(s) < 4 or s[col].nunique() < 2:
            print("  %-42s  (insufficient data)" % label)
            continue
        r = np.corrcoef(s[col], s.pearson)[0, 1]
        rho = s.corr(method="spearman").iloc[0, 1]
        print("  %-42s  r = %+.3f   rho = %+.3f   (n=%d)" % (label, r, rho, len(s)))

    if "label_std" in df.columns and df.label_std.notna().any():
        lo = df[df.label_std < df.label_std.median()]
        hi = df[df.label_std >= df.label_std.median()]
        print("\n  H3 split on median label variability:")
        print("    low-variance  drugs (n=%d): mean PCC %.3f | mean normRMSE %.2f"
              % (len(lo), lo.pearson.mean(), lo.norm_rmse.mean()))
        print("    high-variance drugs (n=%d): mean PCC %.3f | mean normRMSE %.2f"
              % (len(hi), hi.pearson.mean(), hi.norm_rmse.mean()))

    if "moa_covered" in df.columns and df.moa_covered.notna().any():
        cov = df[df.moa_covered == 1]
        unc = df[df.moa_covered == 0]
        if len(cov) and len(unc):
            print("\n  H2 split on mechanistic coverage:")
            print("    MoA covered     (n=%d): mean PCC %.3f" % (len(cov), cov.pearson.mean()))
            print("    MoA not covered (n=%d): mean PCC %.3f" % (len(unc), unc.pearson.mean()))

    df.to_csv(os.path.join(OUT, "lodo_diagnostics.csv"), index=False)
    print("\nWritten to %s/lodo_diagnostics.csv" % OUT)


if __name__ == "__main__":
    main()
