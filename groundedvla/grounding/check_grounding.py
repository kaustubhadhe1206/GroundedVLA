"""Compare computed centroids with simulator ground truth (Step A6/A7 acceptance tests).

Pass criteria: median error < 2 cm and p95 < 4 cm for visible objects; under a 20 degree yaw
the median stays < 2 cm and the robot-frame centroid does not move while the image does.
Ground truth is the center of the object's collision geometry (LiberoWrapper.object_center).

Usage:
    python -m groundedvla.grounding.check_grounding --dataset --tasks T1 T2 T3     # all replayed demos (A6)
    python -m groundedvla.grounding.check_grounding --live --tasks T1 T2 T3        # overlays + yaw test (A6, A7)
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from groundedvla.envs import camera_utils as cu
from groundedvla.envs.libero_wrapper import CameraPerturbation, LiberoWrapper
from groundedvla.grounding.grounding import CameraView
from groundedvla.grounding.masks import instance_id_map
from groundedvla.tasks import TASKS
from groundedvla.utils.paths import load_paths

MEDIAN_LIMIT_CM, P95_LIMIT_CM = 2.0, 4.0
TARGET_COLOR, DEST_COLOR, GT_COLOR = (255, 60, 60), (60, 255, 60), (255, 255, 0)
SIZE = 256


def stats(errors_cm: np.ndarray) -> dict:
    return {"n": int(len(errors_cm)), "median_cm": float(np.median(errors_cm)),
            "p95_cm": float(np.percentile(errors_cm, 95))}


def passed(s: dict) -> bool:
    return s["median_cm"] < MEDIAN_LIMIT_CM and s["p95_cm"] < P95_LIMIT_CM


def check_dataset(paths: dict, tasks: list[str]) -> dict:
    """Stored grounding vs stored ground truth, over every replayed step with the target visible."""
    results = {}
    for key in tasks:
        terr, derr = [], []
        with h5py.File(paths["replayed"] / f"{key}.hdf5") as f:
            for name in f:
                ep = f[name]
                g, gt_t, gt_d = ep["grounding"][()], ep["gt_target"][()], ep["gt_dest"][()]
                visible = g[:, 15] > 0.5
                terr.append(np.linalg.norm(g[visible, :3] - gt_t[visible], axis=1))
                seen = np.any(g[:, 3:6] != 0, axis=1)
                derr.append(np.linalg.norm(g[seen, 3:6] - gt_d[seen], axis=1))
        results[key] = {"target": stats(np.concatenate(terr) * 100), "destination": stats(np.concatenate(derr) * 100)}
        for role, s in results[key].items():
            print(f"{key} {role:11s} n={s['n']:6d} median={s['median_cm']:.2f} cm p95={s['p95_cm']:.2f} cm "
                  f"{'PASS' if passed(s) else 'FAIL'}")
    return results


def overlay(w: LiberoWrapper, obs: dict, view: CameraView, target_id: int, dest_id: int) -> Image.Image:
    """Policy-view frame with the mask tinted in, plus + at computed centers and x at ground truth."""
    img = obs["observation.images.image"].astype(np.float32)
    seg = view.seg[:, ::-1]  # OpenCV order -> policy view
    for inst, color in ((target_id, TARGET_COLOR), (dest_id, DEST_COLOR)):
        img[seg == inst] = 0.5 * img[seg == inst] + 0.5 * np.array(color)
    pil = Image.fromarray(img.astype(np.uint8)).resize((SIZE * 2, SIZE * 2), Image.NEAREST)
    draw = ImageDraw.Draw(pil)
    T_base_to_world = np.linalg.inv(cu.world_to_base(w.sim))

    def to_pixel(p_world):
        rc = cu.project(p_world, w.sim, "agentview", SIZE, SIZE)[0]
        r, c = cu.opencv_to_policy_pixels(rc[0], rc[1], SIZE)
        return float(c) * 2, float(r) * 2

    for inst, name, color in ((target_id, w.goal.target, TARGET_COLOR), (dest_id, w.goal.destination_object, DEST_COLOR)):
        center = view.centroid(inst)
        if center is not None:
            x, y = to_pixel(cu.transform(T_base_to_world, center[None])[0])
            draw.line((x - 7, y, x + 7, y), fill=color, width=2)
            draw.line((x, y - 7, x, y + 7), fill=color, width=2)
        x, y = to_pixel(w.object_center(name))
        draw.line((x - 5, y - 5, x + 5, y + 5), fill=GT_COLOR, width=2)
        draw.line((x - 5, y + 5, x + 5, y - 5), fill=GT_COLOR, width=2)
    # World axes at the robot base: x red, y green, z blue, 10 cm long.
    origin = T_base_to_world[:3, 3]
    for axis, color in zip(np.eye(3), ((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        draw.line((*to_pixel(origin), *to_pixel(origin + 0.1 * axis)), fill=color, width=2)
    return pil


def measure(w: LiberoWrapper, init_state: np.ndarray) -> tuple[dict, CameraView, dict, dict]:
    obs = w.reset(init_state)
    ids = instance_id_map(w.robosuite_env)
    view = CameraView.from_obs(w.sim, obs["raw"])
    world_to_base = cu.world_to_base(w.sim)
    centers, truth = {}, {}
    for role, name in (("target", w.goal.target), ("destination", w.goal.destination_object)):
        centers[role] = view.centroid(ids[name])
        truth[role] = cu.transform(world_to_base, w.object_center(name)[None])[0]
    return obs, view, centers, truth


def check_live(paths: dict, tasks: list[str], episodes: int, yaw: float) -> dict:
    """Overlays, plus the same initial states with and without a yaw: the centers must not move."""
    out_dir = paths["outputs"] / "grounding_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in tasks:
        base = LiberoWrapper(TASKS[key])
        yawed = LiberoWrapper(TASKS[key], camera_perturbation=CameraPerturbation(yaw_deg=yaw))
        ids = instance_id_map(base.robosuite_env)
        target_id, dest_id = ids[base.goal.target], ids[base.goal.destination_object]
        init_states = base.init_states()
        errs = {"target": [], "destination": []}
        errs_yaw = {"target": [], "destination": []}
        shift = {"target": [], "destination": []}
        image_change = []
        for ep in range(episodes):
            obs0, view0, c0, gt0 = measure(base, init_states[ep])
            obs1, view1, c1, gt1 = measure(yawed, init_states[ep])
            for role in errs:
                if c0[role] is not None:
                    errs[role].append(np.linalg.norm(c0[role] - gt0[role]) * 100)
                if c1[role] is not None:
                    errs_yaw[role].append(np.linalg.norm(c1[role] - gt1[role]) * 100)
                if c0[role] is not None and c1[role] is not None:
                    shift[role].append(np.linalg.norm(c1[role] - c0[role]) * 100)
            image_change.append(float(np.abs(obs0["observation.images.image"].astype(float)
                                             - obs1["observation.images.image"].astype(float)).mean()))
            if ep == 0:
                overlay(base, obs0, view0, target_id, dest_id).save(out_dir / f"{key}_overlay.png")
                overlay(yawed, obs1, view1, target_id, dest_id).save(out_dir / f"{key}_overlay_yaw{yaw:+.0f}.png")
        results[key] = {
            role: {"median_cm": float(np.median(errs[role])), "yaw_median_cm": float(np.median(errs_yaw[role])),
                   "yaw_shift_median_cm": float(np.median(shift[role]))} for role in errs
        }
        results[key]["image_change_mean_abs"] = float(np.mean(image_change))
        print(f"{key}: {json.dumps(results[key])}")
        base.close()
        yawed.close()
    print(f"overlays in {out_dir}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="store_true", help="check all replayed demos (needs replay_demos.py)")
    ap.add_argument("--live", action="store_true", help="overlays and the camera yaw test")
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--yaw", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=None, help="save results as JSON")
    a = ap.parse_args()
    p = load_paths()
    report = {}
    if a.dataset:
        report["dataset"] = check_dataset(p, a.tasks)
    if a.live:
        report["live"] = check_live(p, a.tasks, a.episodes, a.yaw)
    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
