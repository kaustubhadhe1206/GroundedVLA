"""Write replayed demos as a LeRobot dataset (Step A5).

One dataset serves all variants; V2/V3 extras are stored as additional features:
    observation.images.image   agentview RGB  (all)
    observation.images.image2  wrist RGB      (all)
    observation.images.mask    R=target, G=destination (V2, V3)
    observation.state          8-D            (all)
    observation.grounding      16-D, normalized (V3)
    action                     7-D
Each variant's config selects which input features the policy uses.

The mask is stored as lossless images, not video: the policy gets exact masks at evaluation time,
and video compression would blur their red and green edges.

Also written next to the dataset:
    grounding_norm.json   workspace center/half-extent, fitted on the training demos only
    splits.json           LeRobot episode indices for the train and val demos of each task

Usage:
    python -m groundedvla.data.convert_to_lerobot --repo-id local/gvla_libero
"""
import argparse
import json
import shutil

import h5py
import numpy as np

from groundedvla.grounding.grounding import GROUNDING_DIM, Normalizer
from groundedvla.tasks import CONTROL_HZ
from groundedvla.utils.paths import REPO_ROOT, load_paths

IMAGE_NAMES = ["height", "width", "channel"]
FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": IMAGE_NAMES},
    "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3), "names": IMAGE_NAMES},
    "observation.images.mask": {"dtype": "image", "shape": (256, 256, 3), "names": IMAGE_NAMES},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "observation.grounding": {"dtype": "float32", "shape": (GROUNDING_DIM,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}


def fit_normalizer(paths: dict, manifest: dict, tasks: list[str]) -> Normalizer:
    vectors = []
    for key in tasks:
        with h5py.File(paths["replayed"] / f"{key}.hdf5") as f:
            vectors += [f[name]["grounding"][()] for name in manifest["demos"][key]["train"]]
    return Normalizer.fit(np.concatenate(vectors))


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/gvla_libero")
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--max-demos-per-task", type=int, default=None, help="small test conversions")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing converted dataset")
    args = ap.parse_args()
    paths = load_paths()
    manifest = json.loads((REPO_ROOT / "groundedvla" / "data" / "split_manifest.json").read_text())

    root = paths["lerobot_datasets"] / args.repo_id.split("/")[-1]
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to rebuild it")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    normalizer = fit_normalizer(paths, manifest, args.tasks)
    ds = LeRobotDataset.create(args.repo_id, fps=CONTROL_HZ, features=FEATURES, root=root, robot_type="panda",
                               image_writer_threads=4)
    splits = {}
    for key in args.tasks:
        splits[key] = {"train": [], "val": []}
        with h5py.File(paths["replayed"] / f"{key}.hdf5") as f:
            instruction = str(f.attrs["instruction"])
            for split in ("train", "val"):
                names = manifest["demos"][key][split]
                for name in names[: args.max_demos_per_task]:
                    ep = f[name]
                    rgb, wrist, mask = ep["agentview_rgb"][()], ep["wrist_rgb"][()], ep["mask"][()]
                    state, action = ep["state"][()], ep["action"][()]
                    grounding = normalizer(ep["grounding"][()])
                    for t in range(len(action)):
                        ds.add_frame({
                            "observation.images.image": rgb[t],
                            "observation.images.image2": wrist[t],
                            "observation.images.mask": mask[t],
                            "observation.state": state[t],
                            "observation.grounding": grounding[t],
                            "action": action[t],
                            "task": instruction,
                        })
                    splits[key][split].append(ds.num_episodes)
                    ds.save_episode()
                    print(f"{key} {split} {name}: episode {splits[key][split][-1]}, {len(action)} frames")
    ds.finalize()
    (root / "grounding_norm.json").write_text(json.dumps(normalizer.state_dict(), indent=2))
    (root / "splits.json").write_text(json.dumps(splits, indent=2))
    print(f"wrote {ds.num_episodes} episodes, {ds.num_frames} frames to {root}")


if __name__ == "__main__":
    main()
