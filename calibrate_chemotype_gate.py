"""
calibrate_chemotype_gate.py
===========================
Test whether a drug's structural similarity to the teacher's training
vocabulary predicts how well the teacher generalises to it.

For each of the 21 O'Neil drugs we already know the leave-one-drug-out (LODO)
Pearson correlation. Here we compute, for each drug, its maximum Morgan
fingerprint Tanimoto similarity to the OTHER 20 drugs -- i.e. how well its
chemotype is covered by the training set when it is held out.

If LODO PCC tracks that similarity, we obtain a calibrated, defensible rule for
deciding which Flow-B (drug-level repurposing) predictions to trust.
"""

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

LODO = "clean_runs/lodo/lodo_per_drug_ALL.csv"
DRUG_MAP = "everything_drug_smiles_map.csv"
OUT = "thesis/aggregate"
os.makedirs(OUT, exist_ok=True)


def fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def main():
    lodo = pd.read_csv(LODO)
    dm = pd.read_csv(DRUG_MAP)
    name2smi = {str(r.drug).lower(): r.smiles for r in dm.itertuples()}

    drugs = lodo.drug.tolist()
    smis = {d: name2smi.get(d.lower()) for d in drugs}
    fps = {d: fp(s) for d, s in smis.items() if s}
    fps = {d: f for d, f in fps.items() if f is not None}
    print("fingerprinted %d / %d drugs" % (len(fps), len(drugs)))

    rows = []
    for d in drugs:
        if d not in fps:
            rows.append({"drug": d, "max_tanimoto_to_others": None,
                         "nearest_drug": None})
            continue
        others = [(o, DataStructs.TanimotoSimilarity(fps[d], fps[o]))
                  for o in fps if o != d]
        others.sort(key=lambda x: -x[1])
        rows.append({
            "drug": d,
            "max_tanimoto_to_others": others[0][1],
            "nearest_drug": others[0][0],
            "second_tanimoto": others[1][1] if len(others) > 1 else None,
        })

    sim = pd.DataFrame(rows)
    df = lodo.merge(sim, on="drug").dropna(subset=["max_tanimoto_to_others"])
    df = df.sort_values("max_tanimoto_to_others", ascending=False)

    r = np.corrcoef(df.max_tanimoto_to_others, df.pearson)[0, 1]
    rho = df[["max_tanimoto_to_others", "pearson"]].corr(method="spearman").iloc[0, 1]

    print()
    print("=" * 92)
    print("  CHEMOTYPE COVERAGE vs LODO GENERALISATION")
    print("=" * 92)
    print(df[["drug", "max_tanimoto_to_others", "nearest_drug",
              "pearson", "rmse"]].to_string(index=False))
    print("-" * 92)
    print("  Pearson  r(max_tanimoto, LODO_pcc) = %+.3f" % r)
    print("  Spearman rho                       = %+.3f" % rho)
    print("=" * 92)

    # Threshold sweep
    print("\n  THRESHOLD SWEEP")
    print("  %-10s %6s %10s %10s %10s" % ("tanimoto", "n", "mean PCC", "min PCC", "n PCC>=0.5"))
    print("  " + "-" * 52)
    best = None
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        s = df[df.max_tanimoto_to_others >= t]
        if len(s) < 2:
            continue
        print("  >= %-7.2f %6d %10.3f %10.3f %10d"
              % (t, len(s), s.pearson.mean(), s.pearson.min(),
                 int((s.pearson >= 0.5).sum())))
        if s.pearson.min() >= 0.3 and (best is None or len(s) > best[1]):
            best = (t, len(s))

    df.to_csv(os.path.join(OUT, "chemotype_gate_calibration.csv"), index=False)

    summary = {
        "pearson_r_tanimoto_vs_lodo": float(r),
        "spearman_rho": float(rho),
        "recommended_threshold": float(best[0]) if best else None,
        "n_drugs_passing": int(best[1]) if best else 0,
    }
    with open(os.path.join(OUT, "chemotype_gate_calibration.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    if best:
        s = df[df.max_tanimoto_to_others >= best[0]]
        print("  RECOMMENDED GATE: nearest-training-drug Tanimoto >= %.2f" % best[0])
        print("    passes: %s" % ", ".join(s.drug.tolist()))
        print("    LODO PCC in that group: %.3f .. %.3f"
              % (s.pearson.min(), s.pearson.max()))
    else:
        print("  No threshold cleanly separates reliable from unreliable drugs.")
    print("\nWritten to %s/chemotype_gate_calibration.csv" % OUT)


if __name__ == "__main__":
    main()
