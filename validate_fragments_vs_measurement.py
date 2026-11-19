#!/usr/bin/env python3
"""
validate_fragments_vs_measurement.py
====================================
Ground-truth test of the SHAP fragment attribution, with no model involved.

The pipeline's foundational claim is that certain fragments harm synergy. So
far that claim has only been checked against the teacher itself -- SHAP values
and leave-one-out deltas both measure how the model's own prediction moves.
This script tests it against measurements.

Design. Every measured combination in dataset_EVERYTHING is expanded into two
(anchor, other) orientations. Observed Loewe scores are then centred within
each (cell line, anchor drug) stratum, which removes cell-line potency and
partner-drug effects and leaves only variation attributable to the other drug.
For a fragment F we compare those residuals between combinations whose other
drug carries F and those whose other drug does not. Membership uses the same
eMolFrag decomposition the model consumes, so "carries F" means exactly what it
means inside the pipeline.

Three fragment sets are evaluated so the result can be read directionally:
SHAP-harmful fragments should show negative residuals, SHAP-helpful fragments
positive ones, and randomly drawn fragments nothing. The random set is the null
that makes the other two interpretable.
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from rdkit import Chem, RDLogger

import synergy_prediction as sp

RDLogger.DisableLog("rdApp.*")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/media/exxact1/magesh/ecomb/dataset_EVERYTHING.pkl")
    p.add_argument("--shap-root",
                   default="clean_runs/clean_runs/2026-07-10_full_rerun_inference/shap")
    p.add_argument("--fragment-map", default="smiles_to_fragments.pkl")
    p.add_argument("--drug-smiles-map", default="everything_drug_smiles_map.csv")
    p.add_argument("--min-carriers", type=int, default=3,
                   help="Minimum distinct carrier drugs before a fragment is tested.")
    p.add_argument("--min-rows", type=int, default=200,
                   help="Minimum measured rows on each side of the comparison.")
    p.add_argument("--n-random", type=int, default=200,
                   help="Size of the random-fragment null set.")
    p.add_argument("--outdir", default="thesis/aggregate")
    p.add_argument("--figdir", default="thesis/figures")
    return p.parse_args()


def collect_shap_fragments(root):
    """Pool per-cell-line SHAP tables into harmful and helpful fragment sets."""
    harmful, helpful = {}, {}
    for f in sorted(glob.glob(os.path.join(root, "*", "top_200_harmful_fragments.csv"))):
        d = pd.read_csv(f)
        if "fragment_smiles" not in d.columns:
            continue
        for r in d.itertuples(index=False):
            s = str(r.fragment_smiles)
            v = float(getattr(r, "fragment_shap_value", np.nan))
            harmful.setdefault(s, []).append(v)
    for f in sorted(glob.glob(os.path.join(root, "*",
                    "top_fragments_across_top_pairs_most_helpful.csv"))):
        d = pd.read_csv(f)
        if "fragment_smiles" not in d.columns:
            continue
        col = ("fragment_shap_value" if "fragment_shap_value" in d.columns
               else d.columns[-1])
        for r in d.itertuples(index=False):
            s = str(r.fragment_smiles)
            try:
                v = float(getattr(r, col))
            except Exception:
                v = np.nan
            helpful.setdefault(s, []).append(v)
    # a fragment flagged both ways is ambiguous; keep it only where it dominates
    both = set(harmful) & set(helpful)
    for s in both:
        if len(harmful[s]) >= len(helpful[s]):
            helpful.pop(s)
        else:
            harmful.pop(s)
    return harmful, helpful


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    harmful, helpful = collect_shap_fragments(args.shap_root)
    print("SHAP fragments pooled: %d harmful, %d helpful" % (len(harmful), len(helpful)))

    fm = sp.load_pickle(Path(args.fragment_map))
    dm = sp.load_drug_smiles_map(Path(args.drug_smiles_map))
    smi_by_name = {str(k).strip().lower(): v for k, v in dm.items()}

    # Membership is resolved by substructure match rather than by exact identity
    # in the eMolFrag output. That output assigns most drugs a single, essentially
    # unique fragment -- no fragment is shared by more than two drugs -- so exact
    # membership cannot support a between-drug comparison. Substructure matching
    # asks the chemically meaningful question instead: do drugs containing this
    # moiety show worse measured synergy than drugs that do not?
    all_smiles = sorted(set(fm.keys()))
    parsed = []
    for s in all_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            parsed.append((s, mol))
    print("parsed %d drug structures for substructure matching" % len(parsed))

    match_cache = {}

    def carriers_of(frag):
        if frag in match_cache:
            return match_cache[frag]
        q = Chem.MolFromSmiles(frag)
        if q is None or q.GetNumHeavyAtoms() < 3:
            match_cache[frag] = set()
            return match_cache[frag]
        hits = {s for s, mol in parsed if mol.HasSubstructMatch(q)}
        match_cache[frag] = hits
        return hits

    print("loading %s ..." % args.dataset, flush=True)
    ds = sp.load_pickle(Path(args.dataset))
    meta = ds["meta"]
    m = meta.assign(
        a=meta.drug_a.astype(str).str.strip(),
        b=meta.drug_b.astype(str).str.strip(),
        c=meta.cell_line_name.astype(str).str.strip(),
        y=meta.y_loewe.astype(float),
    )[["a", "b", "c", "y"]].dropna()

    m["sa"] = m.a.str.lower().map(smi_by_name)
    m["sb"] = m.b.str.lower().map(smi_by_name)
    m = m.dropna(subset=["sa", "sb"])
    m = m[m.sa.isin(fm) & m.sb.isin(fm)]
    print("measured combinations usable: %s" % format(len(m), ","))

    # both orientations: the "other" drug is the one whose fragments we test
    long = pd.concat([
        m.rename(columns={"a": "anchor", "b": "other", "sb": "other_smi"})[
            ["c", "anchor", "other", "other_smi", "y"]],
        m.rename(columns={"b": "anchor", "a": "other", "sa": "other_smi"})[
            ["c", "anchor", "other", "other_smi", "y"]],
    ], ignore_index=True)

    # remove cell-line and anchor-drug effects
    long["resid"] = long.y - long.groupby(["c", "anchor"]).y.transform("mean")
    print("orientation rows: %s | residual sd %.2f"
          % (format(len(long), ","), long.resid.std()))

    smi_arr = long.other_smi.values
    resid = long.resid.values

    def test_fragment(frag):
        carriers = carriers_of(frag)
        if not carriers or len(carriers) < args.min_carriers:
            return None
        mask = np.isin(smi_arr, list(carriers))
        n1 = int(mask.sum())
        n0 = int((~mask).sum())
        if n1 < args.min_rows or n0 < args.min_rows:
            return None
        a, b = resid[mask], resid[~mask]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        return {
            "fragment": frag,
            "n_carrier_drugs": len(carriers),
            "n_rows_carrier": n1,
            "n_rows_other": n0,
            "mean_resid_carrier": float(a.mean()),
            "mean_resid_noncarrier": float(b.mean()),
            "effect": float(a.mean() - b.mean()),
            "cohens_d": float((a.mean() - b.mean()) / pooled) if pooled else np.nan,
            "t": float(t),
            "p": float(p),
        }

    rng = np.random.default_rng(0)
    # Random null: sample candidate fragments and keep those with enough carriers.
    pool = sorted({str(f) for frs in fm.values() for f in (frs or [])})
    random_set = list(rng.choice(pool, size=min(args.n_random * 2, len(pool)),
                                 replace=False))

    results = []
    for label, frags in (("harmful", list(harmful)),
                         ("helpful", list(helpful)),
                         ("random", random_set)):
        got = 0
        for fr in frags:
            r = test_fragment(fr)
            if r:
                r["set"] = label
                if label == "harmful":
                    r["mean_shap"] = float(np.nanmean(harmful[fr]))
                elif label == "helpful":
                    r["mean_shap"] = float(np.nanmean(helpful[fr]))
                else:
                    r["mean_shap"] = np.nan
                results.append(r)
                got += 1
        print("  %-8s tested %d of %d" % (label, got, len(frags)))

    res = pd.DataFrame(results)
    if res.empty:
        raise SystemExit("No fragments met the coverage thresholds.")
    res.to_csv(os.path.join(args.outdir, "fragment_measurement_validation.csv"), index=False)

    print()
    print("=" * 96)
    print("  SHAP FRAGMENTS vs MEASURED SYNERGY  (no model involved)")
    print("=" * 96)
    print("  Residual Loewe after removing cell-line and anchor-drug effects.")
    print("  Negative effect = combinations whose partner carries the fragment score worse.")
    print()
    print("  %-9s %7s %12s %10s %10s %12s" %
          ("set", "n frags", "mean effect", "median", "mean d", "% negative"))
    print("  " + "-" * 66)
    for label in ("harmful", "helpful", "random"):
        s = res[res.set == label]
        if s.empty:
            continue
        print("  %-9s %7d %+12.3f %+10.3f %+10.3f %11.1f%%"
              % (label, len(s), s.effect.mean(), s.effect.median(),
                 s.cohens_d.mean(), 100.0 * (s.effect < 0).mean()))

    h = res[res.set == "harmful"].effect
    r0 = res[res.set == "random"].effect
    hp = res[res.set == "helpful"].effect
    print()
    if len(h) > 2 and len(r0) > 2:
        t, p = stats.ttest_ind(h, r0, equal_var=False)
        u, pu = stats.mannwhitneyu(h, r0, alternative="less")
        print("  harmful vs random : diff %+.3f | t=%+.2f p=%.3g | MWU p=%.3g"
              % (h.mean() - r0.mean(), t, p, pu))
    if len(hp) > 2 and len(r0) > 2:
        t, p = stats.ttest_ind(hp, r0, equal_var=False)
        print("  helpful vs random : diff %+.3f | t=%+.2f p=%.3g"
              % (hp.mean() - r0.mean(), t, p))
    if len(h) > 2 and len(hp) > 2:
        t, p = stats.ttest_ind(h, hp, equal_var=False)
        print("  harmful vs helpful: diff %+.3f | t=%+.2f p=%.3g"
              % (h.mean() - hp.mean(), t, p))

    hh = res[res.set == "harmful"].dropna(subset=["mean_shap"])
    if len(hh) > 3:
        rr = np.corrcoef(hh.mean_shap, hh.effect)[0, 1]
        rho = stats.spearmanr(hh.mean_shap, hh.effect).correlation
        print()
        print("  dose-response: corr(SHAP value, measured effect) r=%+.3f rho=%+.3f (n=%d)"
              % (rr, rho, len(hh)))
        print("  (SHAP is negative for harmful fragments, so a positive r means")
        print("   more negative attribution goes with a more negative measured effect)")

    print()
    print("  STRONGEST MEASURED HARMFUL FRAGMENTS")
    top = res[res.set == "harmful"].nsmallest(15, "effect")
    print(top[["fragment", "n_carrier_drugs", "n_rows_carrier", "effect",
               "cohens_d", "p"]].to_string(index=False, float_format=lambda x: "%.3f" % x))

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    order = ["harmful", "helpful", "random"]
    cols = {"harmful": "#c0392b", "helpful": "#1a7f37", "random": "#888888"}
    data = [res[res.set == s].effect.values for s in order if (res.set == s).any()]
    labs = [s for s in order if (res.set == s).any()]
    bp = ax[0].boxplot(data, labels=labs, patch_artist=True, showfliers=False)
    for patch, s in zip(bp["boxes"], labs):
        patch.set_facecolor(cols[s])
        patch.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2)
    ax[0].axhline(0, color="black", lw=1.2, ls="--")
    ax[0].set_ylabel("measured effect on residual Loewe", fontsize=11, color="black")
    ax[0].set_title("SHAP fragment classes vs measurement", fontsize=12,
                    fontweight="bold", color="black")

    for s in labs:
        ax[1].hist(res[res.set == s].effect, bins=25, alpha=0.6,
                   label=s, color=cols[s], edgecolor="black")
    ax[1].axvline(0, color="black", lw=1.2, ls="--")
    ax[1].set_xlabel("measured effect on residual Loewe", fontsize=11, color="black")
    ax[1].set_ylabel("fragments", fontsize=11, color="black")
    ax[1].set_title("Distribution of measured effects", fontsize=12,
                    fontweight="bold", color="black")
    ax[1].legend(fontsize=10)

    for a in ax:
        a.tick_params(colors="black", labelsize=10)
        for sp_ in a.spines.values():
            sp_.set_color("black")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(args.figdir, "fragment_measurement_validation." + ext),
                    dpi=200, bbox_inches="tight")
    plt.close()

    print()
    print("written -> %s/fragment_measurement_validation.csv" % args.outdir)
    print("figure  -> %s/fragment_measurement_validation.{pdf,png}" % args.figdir)


if __name__ == "__main__":
    main()
