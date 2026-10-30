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
"""
import time

import numpy as np
import torch

from groundedvla.envs.libero_wrapper import GRIPPER_OPEN

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
BASE_RENAME = {"observation.images.image": "observation.images.camera1",
               "observation.images.image2": "observation.images.camera2"}


def to_batch(obs: dict, device: str) -> dict:
    batch = {}
    for k in IMAGE_KEYS:
        img = torch.from_numpy(np.ascontiguousarray(obs[k])).permute(2, 0, 1).float().div_(255.0)
        batch[k] = img.unsqueeze(0).to(device)
    batch["observation.state"] = torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to(device)
    batch["task"] = [obs["task"]]
    return batch


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
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint, config=cfg)
        self.policy.to(self.device).eval()
        overrides = {"device_processor": {"device": self.device}}
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=self.checkpoint, preprocessor_overrides=overrides)
        self.chunk_size = self.policy.config.chunk_size
        self.action_dim = self.policy.config.output_features["action"].shape[0]

    def begin_episode(self, episode_idx: int):
        return None

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def predict_chunk(self, obs: dict) -> np.ndarray:
        batch = to_batch(obs, self.device)
        if self.grounding_fn is not None:  # V3 (Part B)
            t0 = time.perf_counter()
            batch["observation.grounding"] = self.grounding_fn(obs)
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
