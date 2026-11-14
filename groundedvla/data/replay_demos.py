"""Re-render LIBERO demos with masks and 3D grounding (Step A5).

For each demo: reset to its initial state, replay its actions, and save per step: agentview/wrist
RGB, 8-D state, action, the V2 mask image, the raw 16-D grounding vector, and the ground-truth
target/destination centers (for check_grounding.py). Full depth and segmentation are saved only
for a debug subset (plan Section 2.6). No-op steps are removed. Failed replays are dropped and
logged for the split manifest.

Images, masks, depth and segmentation are stored flipped upright, exactly as the policy sees them.
Grounding is computed on the raw arrays before flipping (camera_utils.py explains the conventions).
Finished demos are skipped on restart, so a disconnected Colab session just runs this again.

Usage:
    python -m groundedvla.data.replay_demos --tasks T1 T2 T3
"""
import argparse
import json

import h5py
import numpy as np

from groundedvla.envs import camera_utils as cu
from groundedvla.envs.libero_wrapper import LiberoWrapper, flip
from groundedvla.grounding.grounding import GroundingTracker
from groundedvla.grounding.masks import instance_id_map, mask_image
from groundedvla.tasks import TASKS
from groundedvla.utils.paths import load_paths

NOOP_EPS = 1e-4
DEBUG_DEMOS = 5
IMAGE_KEYS = ("agentview_rgb", "wrist_rgb", "mask", "agentview_depth", "agentview_seg")


def is_noop(action: np.ndarray, prev_gripper: float | None) -> bool:
    """All arm commands ~zero and the gripper command unchanged (same rule as OpenVLA's LIBERO data)."""
    return bool(np.all(np.abs(action[:6]) < NOOP_EPS)) and prev_gripper is not None and action[6] == prev_gripper


def ground_truth_base(w: LiberoWrapper, name: str) -> np.ndarray:
    return cu.transform(cu.world_to_base(w.sim), w.object_center(name)[None])[0].astype(np.float32)


def replay_one(w: LiberoWrapper, demo: h5py.Group, target_id: int, dest_id: int, debug: bool) -> dict | None:
    obs = w.reset(demo["states"][0], settle=False)
    tracker = GroundingTracker(target_id, dest_id)
    keys = ["agentview_rgb", "wrist_rgb", "mask", "state", "action", "grounding", "gt_target", "gt_dest"]
    frames = {k: [] for k in keys + (["agentview_depth", "agentview_seg"] if debug else [])}
    prev_gripper = None
    for action in demo["actions"][()]:
        if is_noop(action, prev_gripper):
            continue
        prev_gripper = action[6]
        frames["agentview_rgb"].append(obs["observation.images.image"])
        frames["wrist_rgb"].append(obs["observation.images.image2"])
        frames["mask"].append(mask_image(obs["seg.agentview"], target_id, dest_id))
        frames["state"].append(obs["observation.state"])
        frames["action"].append(action.astype(np.float32))
        frames["grounding"].append(tracker(w.sim, obs["raw"], normalize=False))
        frames["gt_target"].append(ground_truth_base(w, w.goal.target))
        frames["gt_dest"].append(ground_truth_base(w, w.goal.destination_object))
        if debug:
            frames["agentview_depth"].append(flip(cu.metric_depth(w.sim, obs["raw"]["agentview_depth"]))[..., 0].astype(np.float32))
            frames["agentview_seg"].append(obs["seg.agentview"][..., 0].astype(np.int16))
        obs, _, _, _ = w.step(action)
    if not w.eval_goal():
        return None
    return {k: np.stack(v) for k, v in frames.items()}


def write_demo(f_out: h5py.File, name: str, ep: dict) -> None:
    """All datasets first, then the `complete` flag, so a crash mid-write is redone on restart."""
    g = f_out.require_group(name)
    for k, v in ep.items():
        if k in IMAGE_KEYS:
            g.create_dataset(k, data=v, chunks=(1, *v.shape[1:]), compression="gzip", compression_opts=4)
        else:
            g.create_dataset(k, data=v)
    g.attrs["num_steps"] = len(ep["action"])
    g.attrs["complete"] = True
    f_out.flush()


def save_dropped(path, key: str, names: set) -> None:
    """Update one task's entry, re-reading the file first: tasks may be replayed in parallel."""
    dropped = json.loads(path.read_text()) if path.exists() else {}
    dropped[key] = sorted(names)
    path.write_text(json.dumps(dropped, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--max-demos", type=int, default=50)
    args = ap.parse_args()
    paths = load_paths()
    out_dir = paths["replayed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped_file = out_dir / "dropped_replays.json"
    dropped = json.loads(dropped_file.read_text()) if dropped_file.exists() else {}

    for key in args.tasks:
        spec = TASKS[key]
        src = paths["libero_raw"] / spec.suite / f"{spec.bddl_name}_demo.hdf5"
        w = LiberoWrapper(spec)
        ids = instance_id_map(w.robosuite_env)
        target_id, dest_id = ids[w.goal.target], ids[w.goal.destination_object]
        task_dropped = set(dropped.get(key, []))
        with h5py.File(src) as f_in, h5py.File(out_dir / f"{key}.hdf5", "a") as f_out:
            f_out.attrs["instruction"] = w.instruction
            f_out.attrs["target"], f_out.attrs["destination"] = w.goal.target, w.goal.destination_object
            f_out.attrs["target_id"], f_out.attrs["dest_id"] = target_id, dest_id
            f_out.attrs["instance_ids"] = json.dumps(ids)
            demos = sorted(f_in["data"].keys(), key=lambda s: int(s.split("_")[-1]))[: args.max_demos]
            for i, name in enumerate(demos):
                if name in task_dropped:
                    continue
                if name in f_out:
                    if f_out[name].attrs.get("complete", False):
                        continue
                    del f_out[name]
                ep = replay_one(w, f_in["data"][name], target_id, dest_id, debug=i < DEBUG_DEMOS)
                if ep is None:
                    task_dropped.add(name)
                    save_dropped(dropped_file, key, task_dropped)
                    print(f"{key} {name}: replay failed, dropped")
                    continue
                write_demo(f_out, name, ep)
                print(f"{key} {name}: {len(ep['action'])} steps")
            kept = len(demos) - len(task_dropped)
        save_dropped(dropped_file, key, task_dropped)
        print(f"{key}: kept {kept}/{len(demos)} ({kept / len(demos):.0%})")
        w.close()


if __name__ == "__main__":
    main()
