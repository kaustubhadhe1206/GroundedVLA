"""Write replayed demos as a LeRobot dataset (Step A5).

One dataset serves all variants; V2/V3 extras are stored as additional features:
    observation.images.image        agentview RGB  (all)
    observation.images.image2  wrist RGB      (all)
    observation.images.mask         R=target, G=destination (V2, V3)
    observation.state               8-D            (all)
    observation.grounding           16-D, normalized (V3)
    action                          7-D
Each variant's config selects which input features the policy uses.

Usage:
    python -m groundedvla.data.convert_to_lerobot --repo-id local/gvla_libero
"""
import argparse
import json

import h5py
import numpy as np

from groundedvla.grounding.grounding import GROUNDING_DIM
from groundedvla.grounding.masks import mask_image
from groundedvla.tasks import CONTROL_HZ
from groundedvla.utils.paths import REPO_ROOT, load_paths

FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.mask": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "observation.grounding": {"dtype": "float32", "shape": (GROUNDING_DIM,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}


def grounding_for_episode(ep: h5py.Group) -> np.ndarray:
    """TODO(A6): compute per-step grounding during replay (needs sim for intrinsics/extrinsics)
    and store it in the replay file; read it here instead of zeros."""
    return ep["grounding"][()] if "grounding" in ep else np.zeros((len(ep["action"]), GROUNDING_DIM), np.float32)


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/gvla_libero")
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    args = ap.parse_args()
    paths = load_paths()
    manifest = json.loads((REPO_ROOT / "groundedvla" / "data" / "split_manifest.json").read_text())

    ds = LeRobotDataset.create(args.repo_id, fps=CONTROL_HZ, features=FEATURES,
                               root=paths["lerobot_datasets"] / args.repo_id.split("/")[-1])
    for key in args.tasks:
        with h5py.File(paths["replayed"] / f"{key}.hdf5") as f:
            instruction = f.attrs["instruction"]
            for name in manifest["demos"][key]["train"] + manifest["demos"][key]["val"]:
                ep = f[name]
                g = grounding_for_episode(ep)
                for t in range(len(ep["action"])):
                    ds.add_frame({
                        "observation.images.image": ep["agentview_rgb"][t],
                        "observation.images.image2": ep["wrist_rgb"][t],
                        "observation.images.mask": mask_image(ep["agentview_seg"][t], ep["target_id"][()], ep["dest_id"][()]),
                        "observation.state": ep["state"][t],
                        "observation.grounding": g[t],
                        "action": ep["action"][t],
                        "task": instruction,
                    })
                ds.save_episode()
    ds.finalize() if hasattr(ds, "finalize") else None
    print(f"wrote {ds.num_episodes} episodes to {ds.root}")


if __name__ == "__main__":
    main()
