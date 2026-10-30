"""Fine-tune SmolVLA for V1 / V2 / V3 with one shared loop (Steps B1, B3).

Backbone frozen (freeze_vision_encoder + train_expert_only); V3 additionally trains the
grounding adapter. Checkpoints every `checkpoint_every` steps and resumes from the latest one
after a Colab disconnect.

Usage:
    python -m groundedvla.train.train --config configs/v1_rgb.yaml
    python -m groundedvla.train.train --config configs/v3_grounded.yaml --resume
"""
import argparse
from pathlib import Path

import torch
import yaml

from groundedvla.utils.paths import load_paths


def build_policy(cfg: dict, paths: dict, dataset_meta):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from groundedvla.models.grounded_smolvla import GroundedSmolVLAPolicy

    m = cfg["model"]
    cls = GroundedSmolVLAPolicy if m["variant"] == "v3" else SmolVLAPolicy
    # TODO(B1): load from paths["smolvla_base"]; set config.input_features from
    # m["image_keys"] (+ mask for V2/V3) and dataset_meta; set chunk_size, n_action_steps,
    # freeze_vision_encoder, train_expert_only. Confirm flags in lerobot==0.4.4.
    policy = cls.from_pretrained(str(paths["smolvla_base"]))
    return policy


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return ckpts[-1] if ckpts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = load_paths()
    t = cfg["train"]
    run_dir = paths["checkpoints"] / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(t["seed"])

    # TODO(B1): dataset = LeRobotDataset(cfg["data"]["repo_id"], root=paths["lerobot_datasets"]/...,
    #           episodes=<train ids from split_manifest>, delta_timestamps={"action": [i/20 for i in range(chunk)]})
    # TODO(B1): preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    # TODO(B1): optimizer AdamW(lr, weight_decay) over policy.get_optim_params(); cosine schedule w/ warmup
    # TODO(B1): bf16 autocast loop; log train/val loss, lr, GPU memory to W&B
    # TODO(B1): every checkpoint_every steps -> policy.save_pretrained(run_dir/f"step_{step}") + optimizer/scheduler/
    #           grounding normalizer state; on --resume load latest_checkpoint(run_dir)
    raise NotImplementedError("training loop skeleton; fill in during Step B1")


if __name__ == "__main__":
    main()
