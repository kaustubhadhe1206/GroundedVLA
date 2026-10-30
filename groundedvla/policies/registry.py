"""Build a policy from its name in configs/models.yaml (used by run_sim.py and the eval harness)."""
from pathlib import Path

import yaml

from groundedvla.tasks import TaskSpec
from groundedvla.utils.paths import REPO_ROOT

MODELS_FILE = REPO_ROOT / "configs" / "models.yaml"


def load_registry() -> dict:
    return yaml.safe_load(MODELS_FILE.read_text())


def resolve_weights(hub_id: str, folder: str, paths: dict, download: bool = True) -> Path:
    """Fast local copy first, then the persistent copy (Drive / D:), else download to the persistent copy."""
    for base in (paths["local_models"], paths["models"]):
        p = Path(base) / folder
        if (p / "config.json").exists():
            return p
    dst = Path(paths["models"]) / folder
    if not download:
        raise FileNotFoundError(f"{hub_id} not found in {paths['local_models']} or {paths['models']}")
    from huggingface_hub import snapshot_download

    print(f"[weights] downloading {hub_id} -> {dst}")
    snapshot_download(hub_id, local_dir=dst)
    return dst


def build_policy(name: str, task: TaskSpec, paths: dict, *, checkpoint: str | None = None,
                 device: str | None = None, load_in_4bit: bool = False, load_in_8bit: bool = False):
    registry = load_registry()
    if name not in registry:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(registry)}")
    spec = registry[name]

    if spec["type"] == "replay":
        from groundedvla.policies.replay import ReplayPolicy

        return ReplayPolicy(Path(paths["libero_raw"]) / task.suite / f"{task.bddl_name}_demo.hdf5")

    if spec["type"] == "smolvla":
        from groundedvla.policies.smolvla import SmolVLARunner

        if spec.get("requires_checkpoint") and checkpoint is None:
            raise ValueError(f"{name} needs --checkpoint <dir with config.json + model.safetensors>")
        path = Path(checkpoint) if checkpoint else resolve_weights(spec["hub_id"], spec["folder"], paths)
        return SmolVLARunner(str(path), name=name, device=device, zero_shot_base=spec.get("zero_shot_base", False))

    if spec["type"] == "openvla":
        from groundedvla.policies.openvla import OpenVLARunner

        hub_id = spec["hub_id_per_suite"].get(task.suite)
        if hub_id is None and checkpoint is None:
            raise ValueError(f"no OpenVLA LIBERO checkpoint for suite {task.suite}")
        path = Path(checkpoint) if checkpoint else resolve_weights(hub_id, hub_id.split("/")[-1], paths)
        return OpenVLARunner(str(path), suite=task.suite, name=name, device=device,
                             load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)

    raise ValueError(f"unknown model type {spec['type']!r}")


def default_horizon(name: str) -> int:
    return int(load_registry()[name].get("horizon", 10))
