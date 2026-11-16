"""Export the initial sim states of every validation demo (split manifest) to a small file in the repo.

Validation rollouts (run_eval --split val) only need these states, so with this file Colab doesn't need
the raw LIBERO hdf5 files (GBs) on Drive. Re-run after changing the split manifest or the task list.

    python scripts/export_val_init_states.py
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from groundedvla.eval.run_eval import VAL_STATES_FILE  # noqa: E402
from groundedvla.utils.paths import REPO_ROOT, load_paths  # noqa: E402


def main():
    paths = load_paths()
    manifest = json.loads((REPO_ROOT / "groundedvla" / "data" / "split_manifest.json").read_text())
    out = {}
    for task_key, d in manifest["demos"].items():
        with h5py.File(Path(paths["libero_raw"]) / d["suite"] / f"{d['bddl_name']}_demo.hdf5") as f:
            for name in d["val"]:
                out[f"{task_key}/{name}"] = np.asarray(f["data"][name]["states"][0])
        print(f"{task_key}: {len(d['val'])} val states")
    np.savez_compressed(VAL_STATES_FILE, **out)
    print(f"wrote {VAL_STATES_FILE} ({VAL_STATES_FILE.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
