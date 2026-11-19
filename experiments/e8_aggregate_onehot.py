"""
E8 (aggregate) -- collect the one-hot ablation runs into a table and figure
===========================================================================
Run after the tmux loops finish. Reads whatever
experiments/results/onehot_<ablate>_<split>/ directories exist and reports the
ablation ladder against the full model from E5.

    python experiments/e8_aggregate_onehot.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments/results"
FIG = REPO / "thesis/figures"

FULL = {"cellpair": RES / "teacher_grouped_cellpair/metrics.json",
        "pair": RES / "teacher_grouped_pair/metrics.json"}

LABEL = {"none": "full model (control)",
         "drug": "one-hot drug identity",
         "cell": "one-hot cell identity",
         "both": "one-hot drug + cell"}
ORDER = ["none", "drug", "cell", "both"]
BLUE, GREEN, RED, AMBER = "#1a5fb4", "#1a7f37", "#c0392b", "#b8860b"


def pcc(p):
    d = json.load(open(p))["metrics_test"]
    return d["pearson"], d["rmse"], d["count"]


def main():
    rows = []
    for split in ("cellpair", "pair"):
        if FULL[split].exists():
            r, rm, n = pcc(FULL[split])
            rows.append(dict(split=split, ablate="full (E5)", key="full",
                             pcc=r, rmse=rm, n=n))
        for ab in ORDER:
            f = RES / ("onehot_%s_%s/metrics.json" % (ab, split))
            if f.exists():
                r, rm, n = pcc(f)
                rows.append(dict(split=split, ablate=LABEL[ab], key=ab,
                                 pcc=r, rmse=rm, n=n))
    if not rows:
        raise SystemExit("no ablation runs found yet under %s" % RES)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "e8_onehot_summary.csv", index=False)

    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 78)
    say("  E8  ONE-HOT ABLATION LADDER (grouped splits)")
    say("=" * 78)
    say("")
    for split in ("cellpair", "pair"):
        sub = D[D.split == split]
        if sub.empty:
            continue
        title = ("unseen combination" if split == "cellpair"
                 else "unseen drug pair")
        say("-" * 78)
        say("%s  (--group-mode %s)" % (title, split))
        say("-" * 78)
        say("  %-26s %8s %8s %10s" % ("variant", "PCC", "RMSE", "vs full"))
        say("  " + "-" * 56)
        base = sub[sub.key == "full"].pcc
        base = float(base.iloc[0]) if len(base) else np.nan
        for _, x in sub.iterrows():
            d = "" if x.key == "full" else "%+.3f" % (x.pcc - base)
            say("  %-26s %8.4f %8.3f %10s" % (x.ablate, x.pcc, x.rmse, d))
        say("")
        drug = sub[sub.key == "drug"].pcc
        if len(drug) and np.isfinite(base):
            drop = base - float(drug.iloc[0])
            say("  Replacing the molecular representation with a bare drug")
            say("  identity costs %.3f PCC. %s" % (
                drop,
                "The representation is load-bearing." if drop > 0.10
                else "The representation adds little here."))
        say("")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, split in zip(axes, ("cellpair", "pair")):
        sub = D[D.split == split]
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.set_index("key").reindex(["full"] + ORDER).dropna(subset=["pcc"])
        cols = {"full": BLUE, "none": BLUE, "drug": RED,
                "cell": AMBER, "both": GREEN}
        c = [cols.get(k, "#888888") for k in sub.index]
        ax.bar(range(len(sub)), sub.pcc.values, color=c,
               edgecolor="black", linewidth=1.1)
        for i, v in enumerate(sub.pcc.values):
            ax.text(i, v + 0.012, "%.3f" % v, ha="center",
                    fontsize=10, fontweight="bold", color="black")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([str(a).replace(" (", "\n(").replace(" + ", "\n+ ")
                            for a in sub.ablate], fontsize=9)
        ax.set_title("unseen combination" if split == "cellpair"
                     else "unseen drug pair",
                     fontsize=12, fontweight="bold", color="black")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(colors="black")
        for s in ax.spines.values():
            s.set_color("black")
    axes[0].set_ylabel("Test Pearson correlation", fontsize=12, color="black")
    fig.suptitle("Replacing learned representations with identity codes",
                 fontsize=13, fontweight="bold", color="black")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG / ("onehot_ablation." + ext), dpi=200,
                    bbox_inches="tight")
    plt.close()

    with open(RES / "e8_onehot_summary.txt", "w") as f:
        f.write("\n".join(L) + "\n")
    print("written -> %s/e8_onehot_summary.{txt,csv}" % RES)
    print("figure  -> thesis/figures/onehot_ablation.{pdf,png}")


if __name__ == "__main__":
    main()
