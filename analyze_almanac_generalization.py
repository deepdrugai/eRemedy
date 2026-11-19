"""
analyze_almanac_generalization.py
=================================
Follow-up to the ALMANAC external validation.

The pooled result shows the frozen teacher transfers to ALMANAC on drugs it
knows (PCC 0.47) and collapses on drugs it does not (PCC 0.04). The question
left open is whether some identifiable subset of new chemistry still works --
if so, a gated repurposing claim survives; if not, the negative result is
complete.

With 53 unseen drugs there is finally enough power to test the gate that the
21-drug LODO could not settle: does similarity to the training vocabulary
predict per-drug accuracy? Figures for the chapter are written alongside.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import spearmanr

import synergy_prediction as sp
from pathlib import Path

RDLogger.DisableLog("rdApp.*")

PRED = "clean_runs/external_almanac/almanac_predictions.csv"
OUT = "clean_runs/external_almanac"
FIGS = "thesis/figures"
TRAIN = {
    "391210-10-9", "7-Ethyl-10-hydroxycamptothecin", "915019-65-7", "Bortezomib",
    "Dasatinib", "Dinaciclib", "MK-1775", "MK-2206", "MK-4541", "MRK003",
    "Veliparib", "cyclophosphamide", "etoposide", "geldanamycin", "gemcitabine",
    "metformin", "methotrexate", "mitomycin C", "temozolomide", "vinblastine",
    "vinorelbine",
}
TRAINL = {d.lower() for d in TRAIN}


def fp(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None


def main():
    os.makedirs(FIGS, exist_ok=True)
    df = pd.read_csv(PRED)
    dm = sp.load_drug_smiles_map(Path("everything_drug_smiles_map.csv"))
    low = {str(k).strip().lower(): v for k, v in dm.items()}

    train_fps = [f for f in (fp(low.get(d.lower())) for d in TRAIN) if f is not None]

    # ---- per unseen drug --------------------------------------------------
    rows = []
    unseen = sorted({d for d in set(df.drug_a) | set(df.drug_b)
                     if d.lower() not in TRAINL})
    for d in unseen:
        sub = df[(df.drug_a == d) | (df.drug_b == d)]
        if len(sub) < 30 or sub.y_true.std() == 0 or sub.y_pred.std() == 0:
            continue
        f = fp(low.get(d.lower()))
        sims = [DataStructs.TanimotoSimilarity(f, t) for t in train_fps] if f else []
        rows.append({
            "drug": d,
            "n": len(sub),
            "pearson": np.corrcoef(sub.y_pred, sub.y_true)[0, 1],
            "spearman": spearmanr(sub.y_pred, sub.y_true).correlation,
            "rmse": float(np.sqrt(np.mean((sub.y_pred - sub.y_true) ** 2))),
            "max_tanimoto_to_train": max(sims) if sims else np.nan,
            "nearest_train": (sorted(zip(TRAIN, sims), key=lambda x: -x[1])[0][0]
                              if sims else None),
            "y_true_std": sub.y_true.std(),
        })
    per = pd.DataFrame(rows).sort_values("pearson", ascending=False)
    per.to_csv(os.path.join(OUT, "almanac_per_unseen_drug.csv"), index=False)

    print("=" * 96)
    print("  PER-DRUG ACCURACY, 53 UNSEEN ALMANAC DRUGS")
    print("=" * 96)
    print(per.head(15).to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("  ...")
    print(per.tail(5).to_string(index=False, float_format=lambda x: "%.3f" % x))

    v = per.dropna(subset=["max_tanimoto_to_train"])
    r = np.corrcoef(v.max_tanimoto_to_train, v.pearson)[0, 1]
    rho = spearmanr(v.max_tanimoto_to_train, v.pearson).correlation
    print()
    print("  drugs evaluated                       : %d" % len(per))
    print("  mean per-drug PCC                     : %+.3f" % per.pearson.mean())
    print("  drugs with PCC > 0.3                  : %d" % int((per.pearson > 0.3).sum()))
    print("  drugs with PCC > 0.5                  : %d" % int((per.pearson > 0.5).sum()))
    print()
    print("  SIMILARITY GATE TEST  (n=%d, vs n=21 for LODO)" % len(v))
    print("  corr(max Tanimoto to training, PCC)   : r = %+.3f, rho = %+.3f" % (r, rho))
    for t in (0.2, 0.3, 0.4, 0.5):
        s = v[v.max_tanimoto_to_train >= t]
        if len(s) >= 2:
            print("    Tanimoto >= %.1f : n=%2d  mean PCC %+.3f  max %+.3f"
                  % (t, len(s), s.pearson.mean(), s.pearson.max()))
    if abs(r) < 0.3:
        print("  -> similarity does not identify a reliable subset; no gate available.")

    # ---- figures ----------------------------------------------------------
    strat_order = ["both_known", "one_unseen", "both_unseen"]
    labels = ["both drugs\nknown", "one drug\nunseen", "both drugs\nunseen"]
    colours = ["#1a7f37", "#b8860b", "#c0392b"]

    # Titles are omitted: the caption carries that information, and the panels
    # are sized for a full-width float, so the type is set large.
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))

    pccs = [np.corrcoef(df[df.stratum == s].y_pred, df[df.stratum == s].y_true)[0, 1]
            for s in strat_order]
    ns = [int((df.stratum == s).sum()) for s in strat_order]
    b = ax[0].bar(labels, pccs, color=colours, edgecolor="black", linewidth=1.2)
    for bar, p, n in zip(b, pccs, ns):
        ax[0].text(bar.get_x() + bar.get_width() / 2, p + 0.014,
                   "%.3f\nn=%s" % (p, format(n, ",")), ha="center",
                   fontsize=13, fontweight="bold", color="black")
    ax[0].set_ylabel("Pearson correlation", fontsize=15, color="black")
    ax[0].set_ylim(0, max(pccs) * 1.35)
    ax[0].axhline(0, color="black", lw=0.8)

    ax[1].hist(per.pearson, bins=18, color="#c0392b", edgecolor="black", alpha=0.85)
    ax[1].axvline(0, color="black", lw=1.3, ls="--")
    ax[1].axvline(per.pearson.mean(), color="#1a7f37", lw=2.4,
                  label="mean %.3f" % per.pearson.mean())
    ax[1].set_xlabel("per-drug Pearson correlation", fontsize=15, color="black")
    ax[1].set_ylabel("unseen drugs (n = %d)" % len(per), fontsize=15, color="black")
    ax[1].legend(fontsize=13)

    ax[2].scatter(v.max_tanimoto_to_train, v.pearson, s=70, c="#c0392b",
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    ax[2].axhline(0, color="black", lw=0.9, ls="--")
    ax[2].set_xlabel("max Tanimoto to training vocabulary", fontsize=15, color="black")
    ax[2].set_ylabel("per-drug Pearson correlation", fontsize=15, color="black")
    ax[2].text(0.97, 0.96, "$r = %+.2f$" % r, transform=ax[2].transAxes,
               fontsize=14, fontweight="bold", va="top", ha="right",
               color="black")

    for a in ax:
        a.tick_params(colors="black", labelsize=13)
        for sp_ in a.spines.values():
            sp_.set_color("black")
    plt.tight_layout()
    print("per-drug PCC: n=%d, range %+.3f to %+.3f, median %+.3f; "
          "similarity-vs-PCC r=%+.3f"
          % (len(per), per.pearson.min(), per.pearson.max(),
             per.pearson.median(), r))
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS, "almanac_external_validation." + ext),
                    dpi=200, bbox_inches="tight")
    plt.close()

    # ranking figure
    rk = os.path.join(OUT, "almanac_ranking_contexts.csv")
    if os.path.exists(rk):
        rr = pd.read_csv(rk)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        d = [rr[rr.stratum == s].spearman.dropna() for s in strat_order]
        bp = ax[0].boxplot(d, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(2)
        ax[0].axhline(0, color="black", lw=1.2, ls="--")
        ax[0].set_ylabel("within-context Spearman", fontsize=11, color="black")
        ax[0].set_title("Ranking ability by stratum", fontsize=12,
                        fontweight="bold", color="black")

        x = np.arange(3)
        w = 0.36
        mod = [100 * rr[rr.stratum == s].top1_is_best.mean() for s in strat_order]
        chc = [100 * rr[rr.stratum == s].chance_top1.mean() for s in strat_order]
        ax[1].bar(x - w / 2, mod, w, label="model", color="#1a5fb4",
                  edgecolor="black", linewidth=1.1)
        ax[1].bar(x + w / 2, chc, w, label="chance", color="#999999",
                  edgecolor="black", linewidth=1.1)
        for i, (mv, cv) in enumerate(zip(mod, chc)):
            ax[1].text(i - w / 2, mv + 0.6, "%.1f%%" % mv, ha="center",
                       fontsize=9.5, fontweight="bold", color="black")
            ax[1].text(i + w / 2, cv + 0.6, "%.1f%%" % cv, ha="center",
                       fontsize=9.5, color="black")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels)
        ax[1].set_ylabel("top-1 accuracy (%)", fontsize=11, color="black")
        ax[1].set_title("Picking the best partner", fontsize=12,
                        fontweight="bold", color="black")
        ax[1].legend(fontsize=10)

        for a in ax:
            a.tick_params(colors="black", labelsize=10)
            for sp_ in a.spines.values():
                sp_.set_color("black")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(os.path.join(FIGS, "almanac_ranking." + ext),
                        dpi=200, bbox_inches="tight")
        plt.close()

    print()
    print("figures -> %s/almanac_external_validation.{pdf,png}, almanac_ranking.{pdf,png}" % FIGS)


if __name__ == "__main__":
    main()
