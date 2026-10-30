"""Re-render LIBERO demos with depth + segmentation (Step A5).

For each demo: reset to its initial state, replay its actions, and save per step:
agentview/wrist RGB, metric depth, instance segmentation, 8-D state, action, task/episode
IDs, and target/destination instance IDs. No-op steps are removed; failed replays are
dropped and logged in the split manifest.

Usage:
    python -m groundedvla.data.replay_demos --tasks T1 T2 T3
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from groundedvla.envs import camera_utils as cu
from groundedvla.envs.libero_wrapper import LiberoWrapper
from groundedvla.grounding.masks import instance_ids
from groundedvla.tasks import TASKS
from groundedvla.utils.paths import load_paths

NOOP_EPS = 1e-4


def is_noop(action: np.ndarray, prev_gripper: float | None) -> bool:
    return np.all(np.abs(action[:6]) < NOOP_EPS) and prev_gripper is not None and action[6] == prev_gripper


def replay_one(w: LiberoWrapper, demo: h5py.Group, ids: tuple[int, int]) -> dict | None:
    actions = demo["actions"][()]
    init_state = demo["states"][0]
    w.env.reset()
    raw = w.env.set_init_state(init_state)
    frames = {k: [] for k in ("agentview_rgb", "wrist_rgb", "agentview_depth", "agentview_seg",
                              "state", "action")}
    prev_g = None
    for a in actions:
        if is_noop(a, prev_g):
            continue
        prev_g = a[6]
        obs = w.format_obs(raw)
        frames["agentview_rgb"].append(obs["observation.images.image"])
        frames["wrist_rgb"].append(obs["observation.images.image2"])
        frames["agentview_depth"].append(cu.metric_depth(w.sim, obs["depth.agentview"]).astype(np.float32))
        frames["agentview_seg"].append(obs["seg.agentview"].astype(np.int16))
        frames["state"].append(obs["observation.state"])
        frames["action"].append(a.astype(np.float32))
        raw, _, _, _ = w.env.step(a)
    if not w.env.check_success():
        return None
    out = {k: np.stack(v) for k, v in frames.items()}
    out["target_id"], out["dest_id"] = ids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--max-demos", type=int, default=50)
    args = ap.parse_args()
    paths = load_paths()
    out_dir = paths["replayed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped = {}

    for key in args.tasks:
        spec = TASKS[key]
        src = paths["libero_raw"] / spec.suite / f"{spec.bddl_name}_demo.hdf5"
        w = LiberoWrapper(spec)
        ids = tuple(instance_ids(w.env.env, [w.goal.target, w.goal.destination_object]))
        dropped[key] = []
        with h5py.File(src) as f_in, h5py.File(out_dir / f"{key}.hdf5", "w") as f_out:
            f_out.attrs["instruction"] = w.instruction
            demos = sorted(f_in["data"].keys(), key=lambda s: int(s.split("_")[-1]))[: args.max_demos]
            for name in demos:
                ep = replay_one(w, f_in["data"][name], ids)
                if ep is None:
                    dropped[key].append(name)
                    continue
                g = f_out.create_group(name)
                for k, v in ep.items():
                    g.create_dataset(k, data=v, compression="gzip" if np.ndim(v) > 2 else None)
        print(f"{key}: kept {len(demos) - len(dropped[key])}/{len(demos)}")
        w.close()

    (out_dir / "dropped_replays.json").write_text(json.dumps(dropped, indent=2))


if __name__ == "__main__":
    main()
