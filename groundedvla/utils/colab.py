"""Session setup shared by the Colab notebooks (plan Section 2.6). Keeps notebook cells short.

    from groundedvla.utils.colab import setup_session, stage_to_local
    paths = setup_session()                   # env vars, Drive folders, LIBERO config
    stage_to_local("models/smolvla_libero")   # Drive -> /content/local copy for fast reads
"""
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from groundedvla.utils.paths import REPO_ROOT, load_paths

DRIVE_DIRS = ("models", "datasets", "checkpoints", "outputs")


def setup_session(profile: str = "colab") -> dict:
    os.environ["GVLA_PROFILE"] = profile
    paths = load_paths(profile)
    cache = Path(paths["cache"])
    for k, sub in {"HF_HOME": "huggingface", "HF_LEROBOT_HOME": "lerobot", "TORCH_HOME": "torch"}.items():
        os.environ[k] = str(cache / sub)
    os.environ["LIBERO_CONFIG_PATH"] = str(cache / "libero_config")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    for d in DRIVE_DIRS:
        Path(paths[d]).mkdir(parents=True, exist_ok=True)
    for k in ("local_models", "local_datasets"):
        Path(paths[k]).mkdir(parents=True, exist_ok=True)
    write_libero_config(paths)
    print(f"profile={profile} root={paths['root']} commit={commit_hash()}")
    return paths


def write_libero_config(paths: dict) -> Path:
    """LIBERO asks interactively for its paths on first import unless config.yaml exists."""
    import importlib.util

    pkg = Path(importlib.util.find_spec("libero").submodule_search_locations[0]) / "libero"
    cfg_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"benchmark_root": str(pkg), "bddl_files": str(pkg / "bddl_files"),
           "init_states": str(pkg / "init_files"), "assets": str(pkg / "assets"),
           "datasets": str(paths["libero_raw"])}
    path = cfg_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def stage_to_local(rel: str) -> Path:
    """Copy <Drive root>/<rel> to the matching /content/local folder once per session."""
    paths = load_paths()
    src = Path(paths["root"]) / rel
    top, rest = rel.split("/", 1) if "/" in rel else (rel, "")
    dst = Path(paths[f"local_{top}"]) / rest if f"local_{top}" in paths else Path("/content/local") / rel
    if dst.exists():
        return dst
    if not src.exists():
        raise FileNotFoundError(src)
    print(f"staging {src} -> {dst}")
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"
