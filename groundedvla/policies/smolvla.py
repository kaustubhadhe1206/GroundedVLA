"""SmolVLA through LeRobot (Steps A3, B1-B4): smolvla_base, HuggingFaceVLA/smolvla_libero, our V1-V3.

Inputs follow the Hugging Face LIBERO convention, the same as LeRobot's LiberoProcessorStep:
  observation.images.image   agentview RGB, flipped upright, float [0, 1], (1, 3, 256, 256)
  observation.images.image2  wrist RGB, same format
  observation.state          (1, 8) eef pos, eef axis-angle, gripper finger qpos
  task                       [instruction]
The wrapper already flips the images, so LiberoProcessorStep (which would flip again) is not used.

zero_shot_base=True adapts lerobot/smolvla_base, which expects 3 cameras (camera1..3), a 6-D
state and a 6-D SO-100 action. We feed 2 cameras, the first 6 state values, and map the
*normalized* 6-D output onto LIBERO's 6 arm dimensions (clipped), with the gripper held open.
That is a plumbing test only (plan Section 2.2).

Our checkpoints (V1/V2/V3) say what they were trained with in their train_config.yaml:
  V2, V3   + observation.images.mask (a third camera), from the env (LiberoWrapper with depth/seg)
  V3       + observation.grounding: the env's 16-D `grounding.raw`, normalized with the checkpoint's
             grounding_norm.json, into GroundedSmolVLAPolicy
`grounding_noise_m` (Step B5) adds Gaussian noise to the target and destination centroids before
normalizing; `grounding_enabled = False` hides the grounding tokens (V3 then behaves like V2).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from groundedvla.envs.libero_wrapper import GRIPPER_OPEN

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
MASK_KEY = "observation.images.mask"
GROUNDING_KEY = "observation.grounding"
BASE_RENAME = {"observation.images.image": "observation.images.camera1",
               "observation.images.image2": "observation.images.camera2"}


def to_batch(obs: dict, device: str, image_keys=IMAGE_KEYS) -> dict:
    batch = {}
    for k in image_keys:
        img = torch.from_numpy(np.ascontiguousarray(obs[k])).permute(2, 0, 1).float().div_(255.0)
        batch[k] = img.unsqueeze(0).to(device)
    batch["observation.state"] = torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to(device)
    batch["task"] = [obs["task"]]
    return batch


def noisy_grounding(g: np.ndarray, sigma_m: float, rng: np.random.Generator) -> np.ndarray:
    """Raw 16-D vector with N(0, sigma) noise on the target and destination centroids; the derived
    entries (target - eef, destination - target) are recomputed so the vector stays consistent."""
    g = np.asarray(g, np.float32).copy()
    if sigma_m <= 0:
        return g
    g[0:6] += rng.normal(0.0, sigma_m, 6).astype(np.float32)
    g[9:12] = g[0:3] - g[6:9]
    g[12:15] = g[3:6] - g[0:3]
    return g


def checkpoint_train_config(path) -> dict | None:
    f = Path(path) / "train_config.yaml"
    return yaml.safe_load(f.read_text()) if f.exists() else None


class SmolVLARunner:
    settle = True

    def __init__(self, path: str, name: str = "smolvla", device: str | None = None,
                 zero_shot_base: bool = False, grounding_fn=None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.name, self.checkpoint = name, str(path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.zero_shot_base = zero_shot_base
        self.grounding_fn = grounding_fn
        self.grounding_ms: list[float] = []

        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        cfg.device = self.device
        train_cfg = checkpoint_train_config(self.checkpoint) or {}
        model_cfg = train_cfg.get("model", {})
        self.variant = model_cfg.get("variant")
        self.uses_grounding = bool(model_cfg.get("use_grounding"))
        self.image_keys = tuple(k for k in cfg.input_features if k.startswith("observation.images."))
        if zero_shot_base:
            self.image_keys = IMAGE_KEYS
        self.needs_depth_seg = MASK_KEY in self.image_keys or self.uses_grounding   # env must render seg
        self.grounding_noise_m = 0.0
        self._rng = np.random.default_rng(0)
        if self.uses_grounding:
            from groundedvla.grounding.grounding import Normalizer
            from groundedvla.models.grounded_smolvla import GroundedSmolVLAPolicy

            g = model_cfg.get("grounding", {})
            self.policy = GroundedSmolVLAPolicy.from_pretrained(
                self.checkpoint, config=cfg, num_grounding_tokens=g.get("num_tokens", 2),
                fourier_freqs=g.get("fourier_freqs", 8), mlp_width=g.get("mlp_width", 512))
            self.normalizer = Normalizer.from_state_dict(
                json.loads((Path(self.checkpoint) / "grounding_norm.json").read_text()))
        else:
            self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint, config=cfg)
            self.normalizer = None
        self.policy.to(self.device).eval()
        overrides = {"device_processor": {"device": self.device}}
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=self.checkpoint, preprocessor_overrides=overrides)
        self.chunk_size = self.policy.config.chunk_size
        self.action_dim = self.policy.config.output_features["action"].shape[0]

    def begin_episode(self, episode_idx: int):
        self._rng = np.random.default_rng(episode_idx)       # grounding noise reproducible per episode
        return None

    @property
    def grounding_enabled(self) -> bool:
        return getattr(self.policy, "grounding_enabled", False)

    @grounding_enabled.setter
    def grounding_enabled(self, on: bool) -> None:
        if hasattr(self.policy, "grounding_enabled"):
            self.policy.grounding_enabled = on

    def reset(self) -> None:
        self.policy.reset()
        self.grounding_ms = []                                  # per episode

    @torch.no_grad()
    def predict_chunk(self, obs: dict) -> np.ndarray:
        missing = [k for k in self.image_keys if k not in obs]
        if missing or (self.uses_grounding and "grounding.raw" not in obs):
            raise KeyError(f"{self.name} needs {missing or ['grounding.raw']} from the env: "
                           "create LiberoWrapper(..., with_depth_seg=True)")
        batch = to_batch(obs, self.device, self.image_keys)
        if self.uses_grounding:  # V3
            t0 = time.perf_counter()
            g = self.normalizer(noisy_grounding(obs["grounding.raw"], self.grounding_noise_m, self._rng))
            batch[GROUNDING_KEY] = torch.from_numpy(g).float().unsqueeze(0).to(self.device)
            self.grounding_ms.append(obs.get("perception_ms", 0.0) + (time.perf_counter() - t0) * 1e3)
        elif MASK_KEY in self.image_keys:  # V2: the mask's cost
            self.grounding_ms.append(obs.get("perception_ms", 0.0))
        elif self.grounding_fn is not None:  # custom hook (kept for experiments)
            t0 = time.perf_counter()
            batch[GROUNDING_KEY] = self.grounding_fn(obs)
            self.grounding_ms.append((time.perf_counter() - t0) * 1e3)
        if self.zero_shot_base:
            batch = {BASE_RENAME.get(k, k): v for k, v in batch.items()}
            batch["observation.state"] = batch["observation.state"][:, :6]
        batch = self.pre(batch)
        chunk = self.policy.predict_action_chunk(batch)          # (1, chunk, action_dim), normalized
        if self.zero_shot_base:
            arm = chunk[0, :, :6].float().cpu().clamp(-1, 1).numpy()
            grip = np.full((arm.shape[0], 1), GRIPPER_OPEN, np.float32)
            return np.concatenate([arm, grip], axis=1)
        return self.post(chunk)[0].float().cpu().numpy()          # (chunk, 7) LIBERO actions
