"""Seed pooling and the paired comparison in groundedvla/eval/report.py."""
from groundedvla.eval.report import paired_comparison, summarize


def rec(label, seed, ep, success, task="T1", condition="in_distribution"):
    return {"label": label, "variant": "smolvla_ckpt", "train_seed": seed, "task": task, "condition": condition,
            "episode_idx": ep, "success": success, "final_pos_error_cm": 1.0, "collisions": 0, "steps": 100,
            "correct_object": True, "grasp_success": True, "correct_destination": success, "failure": None}


def test_summary_pools_seeds_per_label():
    rs = [rec("v1_rgb", 0, i, i < 6) for i in range(10)] + [rec("v1_rgb", 1, i, i < 4) for i in range(10)]
    (row,) = summarize(rs)
    assert row["label"] == "v1_rgb" and row["n"] == 20 and row["seeds"] == 2
    assert abs(row["success"] - 0.5) < 1e-9 and abs(row["seed_success_std"] - 0.1) < 1e-9


def test_paired_comparison_matches_episodes():
    a = [rec("v3", 0, i, True) for i in range(20)]
    b = [rec("v1", 0, i, i < 10) for i in range(20)] + [rec("v1", 0, 99, True)]   # ep 99 has no partner
    (c, per_task) = paired_comparison(a + b, "v3", "v1")
    assert c["task"] == "all" and c["pairs"] == 20 and abs(c["diff"] - 0.5) < 1e-9
    assert c["only_a_succeeds"] == 10 and c["only_b_succeeds"] == 0
    assert 0.25 < c["diff_ci95_low"] < 0.5 < c["diff_ci95_high"] <= 0.75
    assert per_task["task"] == "T1"
