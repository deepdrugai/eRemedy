"""
E12 (aggregate) -- did end-to-end training clear the one-hot control?
=====================================================================
Prints the end-to-end teacher against the three arms it has to beat, on the
same grouped splits and seeds. The one-hot column is the decision rule: an
encoder that does not beat a bare drug-identity code is not contributing
chemistry, whatever its absolute accuracy.

    python experiments/e12_aggregate.py
"""

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "experiments/results"
SEEDS = (42, 1, 2, 3)
TARGET = {"cellpair": 0.853, "pair": 0.728}


def pcc(tag):
    p = RES / tag / "metrics.json"
    if not p.exists():
        return None
    return json.load(open(p))["metrics_test"]["pearson"]


def arm(tags):
    v = [x for x in (pcc(t) for t in tags) if x is not None]
    if not v:
        return None
    return np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0), len(v)


def main():
    print("=" * 74)
    print("  E12  END-TO-END ENCODER vs THE CONTROLS  (grouped splits)")
    print("=" * 74)

    verdicts = {}
    for split in ("cellpair", "pair"):
        title = "unseen combination" if split == "cellpair" else "unseen drug pair"
        print("\n%s  (--group-mode %s)" % (title, split))
        print("  %-32s %8s %8s %5s" % ("drug representation", "mean", "sd", "n"))
        print("  " + "-" * 56)

        arms = [
            ("end-to-end GCN (E12)",
             [f"e2e_{split}_s{s}" for s in SEEDS]),
            ("frozen autoencoder (E5/E8)",
             [f"onehot_none_{split}"] + [f"onehot_none_{split}_s{s}" for s in (1, 2, 3)]),
            ("one-hot drug identity (E8)",
             [f"onehot_drug_{split}"] + [f"onehot_drug_{split}_s{s}" for s in (1, 2, 3)]),
            ("ECFP6 + descriptors (E11)",
             [f"ecfp_{split}_s{s}" for s in (42, 1, 2)]),
        ]
        vals = {}
        for name, tags in arms:
            r = arm(tags)
            if r is None:
                print("  %-32s %8s" % (name, "--"))
                continue
            m, sd, n = r
            vals[name] = m
            print("  %-32s %8.4f %8.4f %5d" % (name, m, sd, n))

        e2e = vals.get("end-to-end GCN (E12)")
        oh = vals.get("one-hot drug identity (E8)")
        if e2e is not None and oh is not None:
            d = e2e - oh
            ok = d > 0
            verdicts[split] = ok
            print("\n  end-to-end minus one-hot: %+.4f  ->  %s"
                  % (d, "CLEARS the control" if ok else "does NOT clear the control"))
        elif e2e is None:
            print("\n  (no end-to-end runs found yet)")

    if verdicts:
        print("\n" + "=" * 74)
        if all(verdicts.values()):
            print("  VERDICT: representation is load-bearing on both splits.")
            print("  Flow 1's structural claims are supported; re-run the")
            print("  downstream pipeline and re-tone the paper accordingly.")
        elif any(verdicts.values()):
            print("  VERDICT: mixed. Clears on %s only."
                  % ", ".join(k for k, v in verdicts.items() if v))
            print("  Report as a partial repair; do not restore Flow 1's")
            print("  structural claims without the split that failed.")
        else:
            print("  VERDICT: end-to-end training does not beat drug identity.")
            print("  The finding stands and the paper remains an evaluation")
            print("  contribution. This is a result, not a failed run.")
        print("=" * 74)


if __name__ == "__main__":
    main()
