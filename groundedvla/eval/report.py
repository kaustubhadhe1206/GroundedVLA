"""Result tables, plots, and confidence intervals across V1/V2/V3 (Steps B4, Week 5).

Usage:
    python -m groundedvla.eval.report outputs/eval/*/episodes_smolvla.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

from groundedvla.eval.metrics import wilson_ci


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[(r["variant"], r["task"], r["condition"])].append(r)
    table = {}
    for key, rs in sorted(groups.items()):
        s = sum(r["success"] for r in rs)
        err = np.array([r["final_pos_error_cm"] for r in rs], dtype=float)
        table[key] = {
            "n": len(rs),
            "success": s / len(rs),
            "ci95": wilson_ci(s, len(rs)),
            "pos_err_cm": (np.nanmean(err), np.nanstd(err)),
            "collisions_mean": np.mean([r["collisions"] for r in rs]),
            "failures": {f: sum(r["failure"] == f for r in rs) for f in {r["failure"] for r in rs} if f},
        }
    # TODO(B4): degradation = ID success - OOD success per (variant, task); plots with matplotlib.
    return table


if __name__ == "__main__":
    recs = [r for p in sys.argv[1:] for r in json.load(open(p))]
    for k, v in summarize(recs).items():
        lo, hi = v["ci95"]
        print(f"{k}: n={v['n']} success={v['success']:.2f} [{lo:.2f},{hi:.2f}] failures={v['failures']}")
