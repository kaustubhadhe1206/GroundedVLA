"""Download SmolVLA, LIBERO assets, and the LIBERO demo datasets to the D drive.

Usage (after `. D:\\groundedvla\\activate.ps1`):
    python scripts/download_all.py --what model assets data

All destinations come from configs/paths.yaml. Nothing is written to ~/.cache.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from groundedvla.tasks import TASKS  # noqa: E402
from groundedvla.utils.paths import load_paths  # noqa: E402


def guard_cache_env(paths):
    """Refuse to run if the HF cache would land on C:."""
    os.environ.setdefault("HF_HOME", str(paths["cache"] / "huggingface"))
    if os.environ["HF_HOME"].upper().startswith("C:"):
        sys.exit(f"HF_HOME points to C: ({os.environ['HF_HOME']}). Run activate.ps1 first.")


def download_model(paths):
    from huggingface_hub import snapshot_download

    dst = paths["smolvla_base"]
    print(f"[model] lerobot/smolvla_base -> {dst}")
    snapshot_download("lerobot/smolvla_base", local_dir=dst)


def download_assets(paths):
    from huggingface_hub import snapshot_download

    import libero.libero as ll

    # LIBERO looks for <package>/assets before falling back to ~/.cache, so put them there.
    dst = Path(ll.__file__).parent / "assets"
    print(f"[assets] lerobot/libero-assets -> {dst}")
    snapshot_download("lerobot/libero-assets", repo_type="dataset", local_dir=dst)


def download_data(paths):
    from huggingface_hub import snapshot_download

    dst = paths["libero_raw"]
    # Only the demo files for our tasks (T1-T4), not whole suites.
    files = [f"{t.suite}/{t.bddl_name}_demo.hdf5" for t in TASKS.values()]
    print(f"[data] {files} -> {dst}")
    snapshot_download(
        "yifengzhu-hf/LIBERO-datasets",
        repo_type="dataset",
        local_dir=dst,
        allow_patterns=files,
        max_workers=2,  # keep RAM use low
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", default=["model", "assets", "data"],
                    choices=["model", "assets", "data"])
    args = ap.parse_args()
    paths = load_paths()
    guard_cache_env(paths)
    for what in args.what:
        {"model": download_model, "assets": download_assets, "data": download_data}[what](paths)
    print("done")


if __name__ == "__main__":
    main()
