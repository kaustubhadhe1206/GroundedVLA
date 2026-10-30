"""Step A1 check: load T1, reset, step zero actions, save upright frames from both cameras.

Usage:
    python scripts/smoke_test_env.py --task T1 --steps 20
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio  # noqa: E402
import numpy as np  # noqa: E402

from groundedvla.envs import camera_utils as cu  # noqa: E402
from groundedvla.envs.libero_wrapper import LiberoWrapper  # noqa: E402
from groundedvla.tasks import TASKS  # noqa: E402
from groundedvla.utils.paths import load_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    out = load_paths()["outputs"] / "smoke"
    out.mkdir(parents=True, exist_ok=True)

    w = LiberoWrapper(TASKS[a.task])
    print("instruction:", w.instruction, "| target:", w.goal.target, "| dest:", w.goal.destination_object)
    obs = w.reset(w.init_states()[0])
    for _ in range(a.steps):
        obs, *_ = w.step(w.dummy_action())
    depth_m = cu.metric_depth(w.sim, obs["depth.agentview"])
    seg = obs["seg.agentview"].squeeze(-1)
    imageio.imwrite(out / f"{a.task}_agentview.png", obs["observation.images.image"])
    imageio.imwrite(out / f"{a.task}_wrist.png", obs["observation.images.image2"])
    imageio.imwrite(out / f"{a.task}_agentview_depth.png", to_uint8(depth_m))
    imageio.imwrite(out / f"{a.task}_agentview_seg.png", colorize(seg))
    print("state:", obs["observation.state"].round(3))
    print(f"metric depth range: {depth_m.min():.3f}-{depth_m.max():.3f} m | segmentation ids: {np.unique(seg).tolist()}")
    print("saved frames to", out)
    w.close()


def to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.squeeze(x).astype(np.float32)
    return ((x - x.min()) / max(x.max() - x.min(), 1e-6) * 255).astype(np.uint8)


def colorize(seg: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, (int(seg.max()) + 1, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[seg]


if __name__ == "__main__":
    main()
