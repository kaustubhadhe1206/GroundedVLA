"""Create data/split_manifest.json (plan Section 3.3). All three variants read the same file.

45 train / 5 val demos per task (after dropping failed replays), validation-rollout seeds,
test seeds, and the OOD condition settings from ood/make_ood_states.py.

Usage:
    python -m groundedvla.data.make_split_manifest --seed 0
"""
import argparse
import json

import numpy as np

from groundedvla.tasks import TASKS
from groundedvla.utils.paths import REPO_ROOT, load_paths

MANIFEST = REPO_ROOT / "groundedvla" / "data" / "split_manifest.json"
N_VAL = 5


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
        kept = [f"demo_{i}" for i in range(50) if f"demo_{i}" not in dropped.get(key, [])]
        perm = rng.permutation(kept).tolist()
        manifest["demos"][key] = {"train": sorted(perm[N_VAL:]), "val": sorted(perm[:N_VAL]),
                                  "dropped": dropped.get(key, [])}
    manifest["split_seed"] = args.seed
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
