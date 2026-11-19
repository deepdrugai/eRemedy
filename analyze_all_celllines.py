"""
analyze_all_celllines.py
========================
Aggregate results across all ONEIL cell lines from the inference run:
  symbolic/  rescored/  repurposing/  shap/

Emits a set of CSV tables under thesis/aggregate/ that feed the results chapter.
"""

import json
import os
import pickle
from collections import Counter, defaultdict

import pandas as pd

ROOT = "/home/magesh/ecomb2/clean_runs/clean_runs/2026-07-10_full_rerun_inference"
DATASET = "/media/exxact1/magesh/ecomb/dataset_ONEIL.pkl"
OUT = "/home/magesh/ecomb2/thesis/aggregate"
os.makedirs(OUT, exist_ok=True)

CELL_LINES = sorted(os.listdir(os.path.join(ROOT, "rescored")))


# ── Tissue lookup from the ONEIL metadata ───────────────────────────────────
def tissue_map():
    with open(DATASET, "rb") as f:
        ds = pickle.load(f)
    meta = ds["meta"]
    m = (meta.groupby("cell_line_name")["tissue_name"]
             .agg(lambda s: s.mode().iloc[0]).to_dict())
    n = meta.groupby("cell_line_name").size().to_dict()
    return m, n


TISSUE, N_EXPERIMENTS = tissue_map()


# ── 1. Symbolic regression fit metrics ──────────────────────────────────────
def collect_symbolic():
    rows = []
    for cl in CELL_LINES:
        p = os.path.join(ROOT, "symbolic", cl, "summary.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            s = json.load(f)
        fm = s.get("fit_metrics", {})
        rows.append({
            "cell_line": cl,
            "tissue": TISSUE.get(cl, "?"),
            "n_train_rows": s.get("n_training_rows_after_filter", 0),
            "n_seeds": s.get("n_fragment_seeds", 0),
            "train_pcc": fm.get("train", {}).get("pcc"),
            "train_r2": fm.get("train", {}).get("r2"),
            "val_pcc": fm.get("val", {}).get("pcc"),
            "val_r2": fm.get("val", {}).get("r2"),
            "test_pcc": fm.get("test", {}).get("pcc"),
            "test_r2": fm.get("test", {}).get("r2"),
            "test_rmse": fm.get("test", {}).get("rmse"),
            "equation": s.get("named_equation", ""),
        })
    df = pd.DataFrame(rows).sort_values("test_pcc", ascending=False)
    df.to_csv(os.path.join(OUT, "symbolic_metrics.csv"), index=False)
    return df


# ── 2. Rescoring outcomes ───────────────────────────────────────────────────
def collect_rescored():
    rows, all_hits = [], []
    for cl in CELL_LINES:
        p = os.path.join(ROOT, "rescored", cl, "rescored_reconstructed_candidates.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if d.empty:
            continue
        d["cell_line"] = cl
        all_hits.append(d)

        delta = pd.to_numeric(d["rescored_delta_synergy_vs_model"], errors="coerce").dropna()
        orig = pd.to_numeric(d["teacher_original_synergy"], errors="coerce")
        sel = d.get("selection", pd.Series([""] * len(d)))

        rows.append({
            "cell_line": cl,
            "tissue": TISSUE.get(cl, "?"),
            "n_reconstructed": len(d),
            "n_improved": int((delta > 0).sum()),
            "pct_improved": 100.0 * (delta > 0).sum() / max(len(delta), 1),
            "mean_delta": delta.mean(),
            "median_delta": delta.median(),
            "max_delta": delta.max(),
            "n_delta_gt5": int((delta > 5).sum()),
            "n_delta_gt10": int((delta > 10).sum()),
            "n_delta_gt20": int((delta > 20).sum()),
            "n_rescue": int((sel == "rescue").sum()),
            "n_amplify": int((sel == "amplify").sum()),
            "mean_original_synergy": orig.mean(),
            "n_antagonistic_orig": int((orig < 0).sum()),
        })
    df = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)
    df.to_csv(os.path.join(OUT, "rescored_summary.csv"), index=False)
    pooled = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()
    pooled.to_csv(os.path.join(OUT, "rescored_pooled.csv"), index=False)
    return df, pooled


# ── 3. Repurposing outcomes + MoA ───────────────────────────────────────────
def collect_repurposing():
    rows, all_hits = [], []
    for cl in CELL_LINES:
        p = os.path.join(ROOT, "repurposing", cl, "repurposing_candidates_%s.csv" % cl)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if d.empty:
            rows.append({"cell_line": cl, "tissue": TISSUE.get(cl, "?"),
                         "n_hits": 0, "n_high": 0, "n_medium": 0,
                         "n_low": 0, "n_unknown": 0, "mean_delta": None,
                         "max_delta": None, "n_unique_drugs": 0})
            continue
        d["cell_line"] = cl
        all_hits.append(d)
        conf = d.get("moa_confidence", pd.Series([""] * len(d))).fillna("")
        delta = pd.to_numeric(d["delta_synergy"], errors="coerce")
        rows.append({
            "cell_line": cl,
            "tissue": TISSUE.get(cl, "?"),
            "n_hits": len(d),
            "n_high": int((conf == "high").sum()),
            "n_medium": int((conf == "medium").sum()),
            "n_low": int((conf == "low").sum()),
            "n_unknown": int((conf == "unknown").sum()),
            "mean_delta": delta.mean(),
            "max_delta": delta.max(),
            "n_unique_drugs": d["repurpose_drug_name"].nunique(),
        })
    df = pd.DataFrame(rows).sort_values("n_high", ascending=False)
    df.to_csv(os.path.join(OUT, "repurposing_summary.csv"), index=False)
    pooled = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()
    pooled.to_csv(os.path.join(OUT, "repurposing_pooled.csv"), index=False)
    return df, pooled


# ── 4. Cross-cell-line fragment recurrence ──────────────────────────────────
def cross_fragment(pooled):
    if pooled.empty:
        return pd.DataFrame()
    g = pooled.groupby(["source_fragment_smiles", "candidate_fragment_smiles"]).agg(
        n_cell_lines=("cell_line", "nunique"),
        cell_lines=("cell_line", lambda s: ",".join(sorted(set(s)))),
        n_instances=("cell_line", "size"),
        mean_delta=("rescored_delta_synergy_vs_model", "mean"),
        max_delta=("rescored_delta_synergy_vs_model", "max"),
    ).reset_index().sort_values(["n_cell_lines", "mean_delta"], ascending=False)
    g.to_csv(os.path.join(OUT, "cross_cellline_fragments.csv"), index=False)
    return g


# ── 5. Universal vs specific drug pairs ─────────────────────────────────────
def universal_pairs(pooled):
    if pooled.empty:
        return pd.DataFrame()
    g = pooled.groupby(["drug_a", "drug_b"]).agg(
        n_cell_lines=("cell_line", "nunique"),
        cell_lines=("cell_line", lambda s: ",".join(sorted(set(s)))),
        mean_delta=("rescored_delta_synergy_vs_model", "mean"),
        max_delta=("rescored_delta_synergy_vs_model", "max"),
        mean_original=("teacher_original_synergy", "mean"),
    ).reset_index().sort_values(["n_cell_lines", "mean_delta"], ascending=False)
    g.to_csv(os.path.join(OUT, "universal_pairs.csv"), index=False)
    return g


# ── 6. Most-recurrent harmful fragments from SHAP ───────────────────────────
def harmful_fragments():
    counter = Counter()
    per_cl = defaultdict(list)
    impact = defaultdict(list)
    for cl in CELL_LINES:
        p = os.path.join(ROOT, "shap", cl, "top_200_harmful_fragments.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        col = "fragment_smiles" if "fragment_smiles" in d.columns else d.columns[0]
        val = None
        for c in ["mean_loo_delta", "loo_delta", "mean_attribution", "loo"]:
            if c in d.columns:
                val = c
                break
        for _, r in d.head(50).iterrows():
            frag = r[col]
            counter[frag] += 1
            per_cl[frag].append(cl)
            if val:
                impact[frag].append(r[val])
    rows = [{
        "fragment_smiles": f,
        "n_cell_lines": c,
        "cell_lines": ",".join(sorted(per_cl[f])),
        "mean_impact": (sum(impact[f]) / len(impact[f])) if impact.get(f) else None,
    } for f, c in counter.most_common(200)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "harmful_fragments_recurrent.csv"), index=False)
    return df


# ── 7. Repurposing drug recurrence across cell lines ────────────────────────
def repurpose_drug_recurrence(pooled):
    if pooled.empty:
        return pd.DataFrame()
    g = pooled.groupby("repurpose_drug_name").agg(
        n_cell_lines=("cell_line", "nunique"),
        cell_lines=("cell_line", lambda s: ",".join(sorted(set(s)))),
        n_hits=("cell_line", "size"),
        mean_delta=("delta_synergy", "mean"),
        max_delta=("delta_synergy", "max"),
        best_conf=("moa_confidence",
                   lambda s: min(s, key=lambda x: {"high": 0, "medium": 1,
                                                   "low": 2, "unknown": 3}.get(x, 4))),
    ).reset_index().sort_values(["n_cell_lines", "mean_delta"], ascending=False)
    g.to_csv(os.path.join(OUT, "repurpose_drug_recurrence.csv"), index=False)
    return g


if __name__ == "__main__":
    print("Cell lines:", len(CELL_LINES))

    sym = collect_symbolic()
    print("\n=== SYMBOLIC (top 10 by test PCC) ===")
    print(sym[["cell_line", "tissue", "n_train_rows",
               "train_pcc", "val_pcc", "test_pcc", "test_r2"]].head(10).to_string(index=False))
    print("\ntest_pcc  mean=%.3f  median=%.3f  min=%.3f  max=%.3f"
          % (sym.test_pcc.mean(), sym.test_pcc.median(),
             sym.test_pcc.min(), sym.test_pcc.max()))

    res, res_pool = collect_rescored()
    print("\n=== RESCORED (top 12 by mean delta) ===")
    print(res[["cell_line", "tissue", "n_reconstructed", "pct_improved",
               "mean_delta", "max_delta", "n_delta_gt20"]].head(12).to_string(index=False))
    print("\nTOTAL reconstructed across all lines:", res.n_reconstructed.sum())

    rep, rep_pool = collect_repurposing()
    print("\n=== REPURPOSING (top 12 by n_high) ===")
    print(rep[["cell_line", "tissue", "n_hits", "n_high", "n_medium",
               "mean_delta", "max_delta"]].head(12).to_string(index=False))
    print("\nTOTAL hits:", rep.n_hits.sum(), " high:", rep.n_high.sum(),
          " medium:", rep.n_medium.sum())

    cf = cross_fragment(res_pool)
    print("\n=== CROSS-CELL-LINE FRAGMENT SWAPS (top 12) ===")
    print(cf.head(12).to_string(index=False))

    up = universal_pairs(res_pool)
    print("\n=== UNIVERSAL DRUG PAIRS (top 12) ===")
    print(up.head(12).to_string(index=False))

    hf = harmful_fragments()
    print("\n=== RECURRENT HARMFUL FRAGMENTS (top 12) ===")
    print(hf.head(12).to_string(index=False))

    dr = repurpose_drug_recurrence(rep_pool)
    print("\n=== REPURPOSING DRUG RECURRENCE (top 15) ===")
    print(dr.head(15).to_string(index=False))

    print("\nTables written to", OUT)
