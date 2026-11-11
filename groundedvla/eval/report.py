"""Result tables with confidence intervals across variants, tasks, and conditions (Steps A4, B4).

Reads the per-episode JSONL files written by run_eval.py / run_sim.py and writes a summary CSV.

Rows are per (label, task, condition), where `label` is the variant config (v1_rgb, v3_grounded, ...)
and pools all of its training seeds. `seeds` and `seed_success_std` show how much the success rate
moves between seeds. The Wilson CI treats pooled episodes as independent, so with 2 seeds it is a
little optimistic; the paired comparison below is the main test.

Paired comparison (--compare A B): both variants run the same initial states and seeds, so episodes
are matched on (task, condition, episode_idx, train seed if both have it). The difference in success
is bootstrapped over matched pairs (95% CI), which is far tighter than comparing two separate CIs.

Usage:
    python -m groundedvla.eval.report D:/groundedvla/outputs/eval --out D:/groundedvla/outputs/eval/summary.csv
    python -m groundedvla.eval.report D:/groundedvla/outputs/eval --compare v3_grounded v1_rgb
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from groundedvla.eval.metrics import wilson_ci

BASELINE_CONDITION = "in_distribution"
BOOTSTRAP_SAMPLES = 10_000


def load_records(paths: list[Path]) -> list[dict]:
    files = [f for p in paths for f in (sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])]
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


def label_of(r: dict) -> str:
    return r.get("label") or r["variant"]


def _median(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(values)) if values else float("nan")


def summarize(records: list[dict]) -> list[dict]:
    """One row per (label, task, condition). Latency uses each episode's median and p95."""
    groups = defaultdict(list)
    for r in records:
        groups[(label_of(r), r["task"], r["condition"])].append(r)
    rows = []
    for (label, task, condition), rs in sorted(groups.items()):
        n, s = len(rs), sum(r["success"] for r in rs)
        lo, hi = wilson_ci(s, n)
        err = np.array([r["final_pos_error_cm"] for r in rs], dtype=float)
        collisions = np.array([r["collisions"] for r in rs])
        by_seed = defaultdict(list)
        for r in rs:
            by_seed[r.get("train_seed")].append(r["success"])
        seed_rates = [np.mean(v) for v in by_seed.values()]
        rows.append({
            "label": label, "task": task, "condition": condition, "n": n,
            "seeds": len(by_seed), "success": s / n, "ci95_low": lo, "ci95_high": hi,
            "seed_success_std": float(np.std(seed_rates)) if len(seed_rates) > 1 else float("nan"),
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
    baseline = {(r["label"], r["task"]): r["success"] for r in rows if r["condition"] == BASELINE_CONDITION}
    for r in rows:
        b = baseline.get((r["label"], r["task"]))
        r["degradation"] = b - r["success"] if b is not None else float("nan")


def _pair_key(r: dict, with_seed: bool) -> tuple:
    return (r["task"], r["condition"], r["episode_idx"]) + ((r.get("train_seed"),) if with_seed else ())


def paired_comparison(records: list[dict], a: str, b: str, seed: int = 0) -> list[dict]:
    """Success difference A - B per condition (all tasks pooled) and per (task, condition), on
    matched episodes, with a bootstrap 95% CI and the counts of discordant pairs."""
    ra = [r for r in records if label_of(r) == a]
    rb = [r for r in records if label_of(r) == b]
    with_seed = all(r.get("train_seed") is not None for r in ra + rb)
    sb = {_pair_key(r, with_seed): r["success"] for r in rb}
    pairs = defaultdict(list)
    for r in ra:
        k = _pair_key(r, with_seed)
        if k in sb:
            pairs[(r["task"], r["condition"])].append((r["success"], sb[k]))
            pairs[("all", r["condition"])].append((r["success"], sb[k]))
    rng = np.random.default_rng(seed)
    out = []
    for (task, condition), ps in sorted(pairs.items(), key=lambda kv: (kv[0][1], kv[0][0] != "all", kv[0][0])):
        x = np.array(ps, dtype=float)
        d = x[:, 0] - x[:, 1]
        boot = d[rng.integers(0, len(d), size=(BOOTSTRAP_SAMPLES, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({"a": a, "b": b, "task": task, "condition": condition, "pairs": len(d),
                    "success_a": x[:, 0].mean(), "success_b": x[:, 1].mean(), "diff": d.mean(),
                    "diff_ci95_low": lo, "diff_ci95_high": hi,
                    "only_a_succeeds": int(((x[:, 0] == 1) & (x[:, 1] == 0)).sum()),
                    "only_b_succeeds": int(((x[:, 0] == 0) & (x[:, 1] == 1)).sum())})
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="JSONL files or folders containing them")
    ap.add_argument("--out", type=Path, default=None, help="write the summary as CSV")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                    help="paired success difference A - B on matched episodes (labels, e.g. v3_grounded v1_rgb)")
    args = ap.parse_args()
    records = load_records(args.inputs)
    table = summarize(records)
    for r in table:
        seeds = f" seeds={r['seeds']}" + (f" (sd {r['seed_success_std']:.2f})" if r["seeds"] > 1 else "")
        print(f"{r['label']} {r['task']} {r['condition']}: n={r['n']}{seeds} success={r['success']:.2f} "
              f"[{r['ci95_low']:.2f},{r['ci95_high']:.2f}] grasp={r['grasp']:.2f} err={r['pos_err_cm_mean']:.1f}cm "
              f"collisions={r['collisions_mean']:.2f} failures={r['failures']}")
    if args.out and table:
        write_csv(table, args.out)
    if args.compare:
        comp = paired_comparison(records, *args.compare)
        print(f"\npaired: {args.compare[0]} - {args.compare[1]} (matched episodes, bootstrap 95% CI)")
        for c in comp:
            print(f"  {c['condition']:>16} {c['task']:>4}: {c['success_a']:.2f} vs {c['success_b']:.2f} "
                  f"diff {c['diff']:+.2f} [{c['diff_ci95_low']:+.2f}, {c['diff_ci95_high']:+.2f}] "
                  f"n={c['pairs']} (only A {c['only_a_succeeds']}, only B {c['only_b_succeeds']})")
        if comp and args.out:
            write_csv(comp, args.out.with_name(args.out.stem + f"_paired_{args.compare[0]}_vs_{args.compare[1]}.csv"))
