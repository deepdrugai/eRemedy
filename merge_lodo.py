"""
merge_lodo.py
=============
Merge the sharded leave-one-drug-out results into a single report and
compare drug-level generalisation against the original random-split metric.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

OUTROOT = "clean_runs/lodo"
RANDOM_SPLIT_PCC = 0.9463180736262481   # from oneil_synergy_runs/metrics.json

# Drugs that matter most for the repurposing claims
KEY_DRUGS = {
    "7-Ethyl-10-hydroxycamptothecin": "SN-38 -> irinotecan pairing",
    "915019-65-7": "PI3K/mTOR pairings (XL147, AZD2014, Buparlisib)",
    "metformin": "metformin -> phenformin pairing",
}


def main():
    per_drug = sorted(glob.glob(os.path.join(OUTROOT, "shard_*", "lodo_per_drug.csv")))
    if not per_drug:
        raise SystemExit("No shard results found under %s" % OUTROOT)

    df = pd.concat([pd.read_csv(f) for f in per_drug], ignore_index=True)
    df = df.sort_values("pearson", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(OUTROOT, "lodo_per_drug_ALL.csv"), index=False)

    pooled_files = sorted(glob.glob(
        os.path.join(OUTROOT, "shard_*", "lodo_pooled_predictions.csv")))
    pool = pd.concat([pd.read_csv(f) for f in pooled_files], ignore_index=True)
    pool.to_csv(os.path.join(OUTROOT, "lodo_pooled_ALL.csv"), index=False)

    yt, yp = pool.y_true.values, pool.y_pred.values
    pooled_pcc = float(np.corrcoef(yt, yp)[0, 1])
    pooled_rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    summary = {
        "n_drugs": int(len(df)),
        "n_predictions": int(len(pool)),
        "pooled_pearson": pooled_pcc,
        "pooled_rmse": pooled_rmse,
        "pooled_mae": float(np.mean(np.abs(yt - yp))),
        "mean_per_drug_pearson": float(df.pearson.mean()),
        "median_per_drug_pearson": float(df.pearson.median()),
        "min_per_drug_pearson": float(df.pearson.min()),
        "max_per_drug_pearson": float(df.pearson.max()),
        "n_drugs_pcc_above_0.7": int((df.pearson >= 0.7).sum()),
        "n_drugs_pcc_above_0.5": int((df.pearson >= 0.5).sum()),
        "n_drugs_pcc_below_0.3": int((df.pearson < 0.3).sum()),
        "n_drugs_pcc_negative": int((df.pearson < 0).sum()),
        "random_split_reference_pearson": RANDOM_SPLIT_PCC,
        "degradation": RANDOM_SPLIT_PCC - pooled_pcc,
    }
    with open(os.path.join(OUTROOT, "lodo_summary_ALL.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 78)
    print("  LEAVE-ONE-DRUG-OUT VALIDATION  --  MERGED RESULTS")
    print("=" * 78)
    print(df[["drug", "n_test", "pearson", "spearman", "rmse", "mae"]]
          .to_string(index=False))
    print("-" * 78)
    print("  pooled PCC (drug-level split) : %.4f" % pooled_pcc)
    print("  pooled RMSE                   : %.3f" % pooled_rmse)
    print("  mean / median per-drug PCC    : %.4f / %.4f"
          % (summary["mean_per_drug_pearson"], summary["median_per_drug_pearson"]))
    print("  range                         : %.3f .. %.3f"
          % (summary["min_per_drug_pearson"], summary["max_per_drug_pearson"]))
    print()
    print("  reference (random split)      : %.4f" % RANDOM_SPLIT_PCC)
    print("  DEGRADATION                   : %.4f" % summary["degradation"])
    print("-" * 78)
    print("  drugs with PCC >= 0.7 : %2d / %d" % (summary["n_drugs_pcc_above_0.7"], len(df)))
    print("  drugs with PCC >= 0.5 : %2d / %d" % (summary["n_drugs_pcc_above_0.5"], len(df)))
    print("  drugs with PCC <  0.3 : %2d / %d" % (summary["n_drugs_pcc_below_0.3"], len(df)))
    print("  drugs with PCC <  0   : %2d / %d" % (summary["n_drugs_pcc_negative"], len(df)))
    print("=" * 78)

    print("\n  KEY DRUGS FOR THE REPURPOSING CLAIMS")
    print("  " + "-" * 74)
    for drug, why in KEY_DRUGS.items():
        row = df[df.drug == drug]
        if row.empty:
            print("    %-34s NOT EVALUATED" % drug)
            continue
        r = row.iloc[0]
        verdict = ("supports drug-level claims" if r.pearson >= 0.7 else
                   "weak -- treat drug-level claims cautiously" if r.pearson >= 0.5 else
                   "DOES NOT support drug-level claims")
        print("    %-34s PCC=%+.3f  RMSE=%5.2f  -> %s"
              % (drug, r.pearson, r.rmse, verdict))
        print("      (%s)" % why)

    print("\n  INTERPRETATION")
    print("  " + "-" * 74)
    if pooled_pcc >= 0.7:
        print("    Teacher generalises well to unseen drugs. Flow B (scoring")
        print("    repurposed drugs directly) is defensible.")
    elif pooled_pcc >= 0.5:
        print("    Moderate generalisation. Flow B is usable but drug-level")
        print("    predictions carry meaningful uncertainty -- report LODO")
        print("    alongside every repurposing number.")
    else:
        print("    Poor generalisation to unseen drugs. Flow B predictions for")
        print("    drugs outside the 21-drug training vocabulary are NOT")
        print("    reliable. Options: (a) restrict repurposing to the 21 known")
        print("    drugs, (b) retrain the teacher on a larger drug vocabulary")
        print("    (e.g. dataset_EVERYTHING), or (c) present Flow B as")
        print("    hypothesis generation only, with no quantitative claims.")
    print()
    print("Written to %s/lodo_per_drug_ALL.csv and lodo_summary_ALL.json" % OUTROOT)


if __name__ == "__main__":
    main()
