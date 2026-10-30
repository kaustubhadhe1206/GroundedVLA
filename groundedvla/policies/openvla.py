"""OpenVLA-7B LIBERO checkpoints (Step A3 pipeline check). Never fine-tuned by us.

Mirrors openvla/experiments/robot/libero/run_libero_eval.py:
- agentview image rotated 180 deg (our wrapper already does this), JPEG round-trip, resize to 224
- center crop to 90% of the area, resize back to 224 (the LIBERO checkpoints were trained with crops)
- prompt "In: What action should the robot take to {task}?\nOut:"
- unnorm_key = suite name, or "<suite>_no_noops" if that is what the checkpoint stores
- gripper: [0, 1] -> [-1, +1], binarize, then flip the sign (OpenVLA's dataloader convention)

Runs only in the OpenVLA environment (transformers==4.40.1, see requirements-openvla.txt) and
needs ~16 GB of GPU memory in bf16 (Colab A100; a T4 needs load_in_4bit / load_in_8bit).
"""
import importlib.util
import io
import math

import numpy as np
import torch
from PIL import Image

RESIZE = 224
CROP_SCALE = 0.9


def openvla_image(rgb: np.ndarray, center_crop: bool = True) -> Image.Image:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    img = Image.open(buf).convert("RGB").resize((RESIZE, RESIZE), Image.LANCZOS)
    if center_crop:
        side = math.sqrt(CROP_SCALE) * RESIZE
        lo = (RESIZE - side) / 2
        img = img.crop((lo, lo, lo + side, lo + side)).resize((RESIZE, RESIZE), Image.BILINEAR)
    return img


class OpenVLARunner:
    settle = True

    def __init__(self, path: str, suite: str, name: str = "openvla_7b", device: str | None = None,
                 load_in_4bit: bool = False, load_in_8bit: bool = False, center_crop: bool = True):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.name, self.checkpoint = name, str(path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.center_crop = center_crop
        kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
        if importlib.util.find_spec("flash_attn") is not None:
            kwargs["attn_implementation"] = "flash_attention_2"
        quantized = load_in_4bit or load_in_8bit
        if quantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(self.checkpoint, **kwargs)
        if not quantized:
            self.vla = self.vla.to(self.device)
        self.vla.eval()

        stats = self.vla.norm_stats
        self.unnorm_key = suite if suite in stats else f"{suite}_no_noops"
        if self.unnorm_key not in stats:
            raise KeyError(f"{suite!r} not in checkpoint norm_stats {list(stats)}: wrong checkpoint for this suite?")

    def begin_episode(self, episode_idx: int):
        return None

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def predict_chunk(self, obs: dict) -> np.ndarray:
        image = openvla_image(obs["observation.images.image"], self.center_crop)
        prompt = f"In: What action should the robot take to {obs['task'].lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        action = np.asarray(self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False),
                            dtype=np.float32)
        action[-1] = -np.sign(2.0 * action[-1] - 1.0)
        return action[None]
