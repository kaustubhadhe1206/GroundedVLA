"""Result tables with confidence intervals across variants, tasks, and conditions (Steps A4, B4).

Reads the per-episode JSONL files written by run_eval.py / run_sim.py and writes a summary CSV.

Usage:
    python -m groundedvla.eval.report D:/groundedvla/outputs/eval --out D:/groundedvla/outputs/eval/summary.csv
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from groundedvla.eval.metrics import wilson_ci

BASELINE_CONDITION = "in_distribution"


def load_records(paths: list[Path]) -> list[dict]:
    files = [f for p in paths for f in (sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])]
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


def _median(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(values)) if values else float("nan")


def summarize(records: list[dict]) -> list[dict]:
    """One row per (variant, task, condition). Latency uses each episode's median and p95."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["variant"], r["task"], r["condition"])].append(r)
    rows = []
    for (variant, task, condition), rs in sorted(groups.items()):
        n, s = len(rs), sum(r["success"] for r in rs)
        lo, hi = wilson_ci(s, n)
        err = np.array([r["final_pos_error_cm"] for r in rs], dtype=float)
        collisions = np.array([r["collisions"] for r in rs])
        rows.append({
            "variant": variant, "task": task, "condition": condition, "n": n,
            "success": s / n, "ci95_low": lo, "ci95_high": hi,
            "correct_object": np.mean([r["correct_object"] for r in rs]),
            "grasp": np.mean([r["grasp_success"] for r in rs]),
            "correct_destination": np.mean([r["correct_destination"] for r in rs]),
            "pos_err_cm_mean": np.nanmean(err), "pos_err_cm_std": np.nanstd(err),
            "collisions_mean": collisions.mean(), "pct_episodes_with_collision": 100 * np.mean(collisions > 0),
            "steps_mean": np.mean([r["steps"] for r in rs]),
            "latency_ms_median": _median(rs, "latency_ms_median"), "latency_ms_p95": _median(rs, "latency_ms_p95"),
            "grounding_ms_median": _median(rs, "grounding_ms_median"), "grounding_ms_p95": _median(rs, "grounding_ms_p95"),
            "failures": json.dumps(Counter(r["failure"] for r in rs if r["failure"])),
        })
    add_degradation(rows)
    return rows


def add_degradation(rows: list[dict]) -> None:
    """Degradation = in-distribution success minus this condition's success (plan Section 6.6)."""
    baseline = {(r["variant"], r["task"]): r["success"] for r in rows if r["condition"] == BASELINE_CONDITION}
    for r in rows:
        b = baseline.get((r["variant"], r["task"]))
        r["degradation"] = b - r["success"] if b is not None else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="JSONL files or folders containing them")
    ap.add_argument("--out", type=Path, default=None, help="write the summary as CSV")
    args = ap.parse_args()
    table = summarize(load_records(args.inputs))
    for r in table:
        print(f"{r['variant']} {r['task']} {r['condition']}: n={r['n']} success={r['success']:.2f} "
              f"[{r['ci95_low']:.2f},{r['ci95_high']:.2f}] grasp={r['grasp']:.2f} err={r['pos_err_cm_mean']:.1f}cm "
              f"collisions={r['collisions_mean']:.2f} failures={r['failures']}")
    if args.out and table:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"wrote {args.out}")
