"""Resolve project paths from configs/paths/<profile>.yaml so nothing is hard-coded (Section 2.5, 2.6).

Profiles:
  local  - laptop, everything under D:/groundedvla
  colab  - persistent files on Google Drive, fast working copies on the Colab local disk
The profile is picked with the GVLA_PROFILE env var, else auto-detected (Colab -> colab, otherwise local).
GVLA_ROOT overrides the profile's `root`.
"""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS_DIR = REPO_ROOT / "configs" / "paths"


def detect_profile() -> str:
    if os.environ.get("GVLA_PROFILE"):
        return os.environ["GVLA_PROFILE"]
    return "colab" if Path("/content").is_dir() and "COLAB_RELEASE_TAG" in os.environ else "local"


def load_paths(profile: str | None = None) -> dict:
    profile = profile or detect_profile()
    with open(PATHS_DIR / f"{profile}.yaml") as f:
        cfg = yaml.safe_load(f)
    root = Path(os.environ.get("GVLA_ROOT", cfg.pop("root")))
    paths = {"root": root, "profile": profile}
    for key, rel in cfg.items():
        p = Path(rel)
        paths[key] = p if p.is_absolute() else root / p
    return paths
