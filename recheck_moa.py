"""
Recompute MoA confidence on the pooled repurposing table using the corrected
ATC level-1 parsing (strip whitespace before taking the first character).

Read-only with respect to the pipeline outputs: writes only to thesis/aggregate/.
"""
import csv
import os

import pandas as pd

OUT = "/home/magesh/ecomb2/thesis/aggregate"
CACHE = "/home/magesh/ecomb2/drug_moa_cache.csv"

cache = {}
with open(CACHE, newline="") as f:
    for row in csv.DictReader(f):
        cache[row["drug_name"].lower()] = row


def atc_l1(row):
    codes = (row.get("atc_classifications") or "")
    if codes.lower() == "nan":
        return set()
    return {c.strip()[0].upper() for c in codes.split(",") if c.strip()}


def targets(row):
    t = (row.get("target_chembl_ids") or "")
    if t.lower() == "nan":
        return set()
    return {x.strip() for x in t.split(";") if x.strip()}


def confidence(a_name, b_name):
    a = cache.get(str(a_name).lower(), {})
    b = cache.get(str(b_name).lower(), {})
    if not a or not b:
        return "unknown", False, False
    tm = bool(targets(a) and targets(b) and targets(a) & targets(b))
    am = bool(atc_l1(a) and atc_l1(b) and atc_l1(a) & atc_l1(b))
    if tm:
        return "high", tm, am
    if am:
        return "medium", tm, am
    return "low", tm, am


df = pd.read_csv(os.path.join(OUT, "repurposing_pooled.csv"))
new = [confidence(r.replaced_drug, r.repurpose_drug_name) for r in df.itertuples()]
df["moa_confidence_fixed"] = [x[0] for x in new]
df["moa_match_target_fixed"] = [x[1] for x in new]
df["moa_match_atc_fixed"] = [x[2] for x in new]
df.to_csv(os.path.join(OUT, "repurposing_pooled_fixed.csv"), index=False)

print("=== CONFIDENCE: before vs after ATC fix ===")
comp = pd.crosstab(df.moa_confidence, df.moa_confidence_fixed,
                   rownames=["before"], colnames=["after"])
print(comp.to_string())

print("\n=== Counts ===")
print("before:", df.moa_confidence.value_counts().to_dict())
print("after :", df.moa_confidence_fixed.value_counts().to_dict())

print("\n=== MEDIUM pairings AFTER fix ===")
med = df[df.moa_confidence_fixed == "medium"]
if len(med):
    g = med.groupby(["replaced_drug", "repurpose_drug_name"]).agg(
        n_cell_lines=("cell_line", "nunique"),
        mean_delta=("delta_synergy", "mean"),
        max_delta=("delta_synergy", "max"),
        replaced_atc=("replaced_atc", "first"),
        repurpose_atc=("repurpose_atc", "first"),
    ).reset_index().sort_values("n_cell_lines", ascending=False)
    print(g.to_string(index=False))
    g.to_csv(os.path.join(OUT, "moa_medium_pairings_fixed.csv"), index=False)
else:
    print("  none")

print("\n=== HIGH pairings (unchanged by fix) ===")
hi = df[df.moa_confidence_fixed == "high"]
g = hi.groupby(["replaced_drug", "repurpose_drug_name"]).agg(
    n_cell_lines=("cell_line", "nunique"),
    cell_lines=("cell_line", lambda s: ",".join(sorted(set(s)))),
    n_hits=("cell_line", "size"),
    mean_delta=("delta_synergy", "mean"),
    max_delta=("delta_synergy", "max"),
    moa=("repurpose_moa", "first"),
).reset_index().sort_values(["n_cell_lines", "mean_delta"], ascending=False)
print(g.to_string(index=False))
g.to_csv(os.path.join(OUT, "moa_high_pairings_fixed.csv"), index=False)

# Per-cell-line corrected counts
per = df.groupby("cell_line").moa_confidence_fixed.value_counts().unstack(fill_value=0)
for c in ["high", "medium", "low", "unknown"]:
    if c not in per.columns:
        per[c] = 0
per = per[["high", "medium", "low", "unknown"]].reset_index()
per["total"] = per[["high", "medium", "low", "unknown"]].sum(axis=1)
per.to_csv(os.path.join(OUT, "repurposing_moa_per_cellline_fixed.csv"), index=False)
print("\n=== PER-CELL-LINE (corrected) top 15 by high ===")
print(per.sort_values(["high", "medium"], ascending=False).head(15).to_string(index=False))

print("\nWritten to", OUT)
