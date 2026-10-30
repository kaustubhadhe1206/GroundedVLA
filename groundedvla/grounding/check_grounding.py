"""Compare computed centroids with simulator ground truth (Step A6/A7 acceptance tests).

Pass criteria: median error < 2 cm and p95 < 4 cm for visible objects;
under a 20 degree yaw perturbation the median stays < 2 cm.

Usage:
    python -m groundedvla.grounding.check_grounding --task T1 --episodes 5 [--yaw 20]
"""
import argparse

import numpy as np

from groundedvla.envs import camera_utils as cu
from groundedvla.envs.libero_wrapper import CameraPerturbation, LiberoWrapper
from groundedvla.grounding.grounding import object_centroid
from groundedvla.grounding.masks import instance_ids
from groundedvla.tasks import TASKS


def gt_position_base(sim, body_name: str) -> np.ndarray:
    # TODO(A6): LIBERO body names may carry a "_main" suffix; resolve from env.objects_dict.
    pos_w = sim.data.body_xpos[sim.model.body_name2id(body_name)]
    return cu.transform(cu.world_to_base(sim), pos_w[None])[0]


def run(task_key: str, episodes: int, steps: int, yaw: float):
    pert = CameraPerturbation(yaw_deg=yaw) if yaw else None
    w = LiberoWrapper(TASKS[task_key], camera_perturbation=pert)
    tid, did = instance_ids(w.env.env, [w.goal.target, w.goal.destination_object])
    errs = []
    for ep in range(episodes):
        obs = w.reset()
        for _ in range(steps):
            c = object_centroid(w.sim, obs["raw"], tid)
            if c is not None:
                errs.append(np.linalg.norm(c - gt_position_base(w.sim, w.goal.target)))
            obs, *_ = w.step(w.dummy_action())
    errs = np.array(errs) * 100
    print(f"{task_key} yaw={yaw}: n={len(errs)} median={np.median(errs):.2f} cm p95={np.percentile(errs, 95):.2f} cm")
    # TODO(A6): save overlay frames (masks, centroids, projected GT, axes) to outputs/grounding_checks
    w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--yaw", type=float, default=0.0)
    a = ap.parse_args()
    run(a.task, a.episodes, a.steps, a.yaw)
