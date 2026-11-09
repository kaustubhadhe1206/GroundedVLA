"""Fill data/split_manifest.json (plan Section 3.3). All three variants read the same file.

5 validation demos per task, the rest for training, chosen from the demos that replayed
successfully in replay_demos.py. Dropped replays are recorded too.

Usage:
    python -m groundedvla.data.make_split_manifest --seed 0
"""
import argparse
import json

import h5py
import numpy as np

from groundedvla.tasks import TASKS
from groundedvla.utils.paths import REPO_ROOT, load_paths

MANIFEST = REPO_ROOT / "groundedvla" / "data" / "split_manifest.json"
N_VAL = 5


def demo_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    args = ap.parse_args()
    paths = load_paths()
    dropped_file = paths["replayed"] / "dropped_replays.json"
    dropped = json.loads(dropped_file.read_text()) if dropped_file.exists() else {}

    rng = np.random.default_rng(args.seed)
    manifest = json.loads(MANIFEST.read_text())
    for key in args.tasks:
        spec = TASKS[key]
        with h5py.File(paths["replayed"] / f"{key}.hdf5") as f:
            kept = sorted((n for n in f if f[n].attrs.get("complete", False)), key=demo_sort_key)
        perm = rng.permutation(kept).tolist()
        manifest["demos"][key] = {"suite": spec.suite, "bddl_name": spec.bddl_name,
                                  "train": sorted(perm[N_VAL:], key=demo_sort_key),
                                  "val": sorted(perm[:N_VAL], key=demo_sort_key),
                                  "dropped": dropped.get(key, [])}
        print(f"{key}: {len(perm) - N_VAL} train, {N_VAL} val, {len(dropped.get(key, []))} dropped")
    manifest["split_seed"] = args.seed
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
