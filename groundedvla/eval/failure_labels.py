"""Automatic failure categories (Step A4). First matching label wins, in this order."""
from dataclasses import dataclass

import numpy as np

FAILURE_ORDER = ["wrong_object", "collision", "failed_grasp", "dropped_object",
                 "wrong_destination", "motion_failure", "timeout"]


@dataclass
class EpisodeTrace:
    """Signals collected by rollout.py during an episode."""
    first_lifted: str | None            # name of first object lifted, or None
    target: str
    max_nontarget_displacement_m: float
    target_knocked_over_before_grasp: bool
    target_ever_lifted: bool
    target_on_table_outside_dest: bool
    target_resting_on: str | None       # object the target ended on/in, if any
    destination: str
    eef_positions: np.ndarray            # (T, 3)
    hit_joint_limit: bool


def label_failure(tr: EpisodeTrace) -> str:
    if tr.first_lifted is not None and tr.first_lifted != tr.target:
        return "wrong_object"
    if tr.max_nontarget_displacement_m > 0.05 or tr.target_knocked_over_before_grasp:
        return "collision"
    if not tr.target_ever_lifted:
        return "failed_grasp"
    if tr.target_on_table_outside_dest:
        return "dropped_object"
    if tr.target_resting_on is not None and tr.target_resting_on != tr.destination:
        return "wrong_destination"
    tail = tr.eef_positions[-50:]
    if tr.hit_joint_limit or (len(tail) == 50 and np.linalg.norm(tail.max(0) - tail.min(0)) < 0.01):
        return "motion_failure"
    return "timeout"
