"""Check the converted dataset against SmolVLA's expectations (Step A5 done-when).

Loads the dataset the way training will (50-step action chunks), saves a grid of random frames,
prints value ranges, and runs a real batch through the preprocessor and the loss of the already
fine-tuned HuggingFaceVLA/smolvla_libero. A low loss on our data means images, tokenization, state
format, action normalization and chunk dimensions all match what SmolVLA expects.

Usage:
    python -m groundedvla.data.check_dataset --repo-id local/gvla_libero
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from groundedvla.policies.registry import load_registry, resolve_weights
from groundedvla.tasks import CONTROL_HZ
from groundedvla.utils.paths import load_paths

CHUNK_SIZE = 50
GRID_FRAMES = 6
RGB_KEYS = ("observation.images.image", "observation.images.image2")
MASK_KEY = "observation.images.mask"


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def save_grid(ds, path) -> None:
    idx = np.random.default_rng(0).choice(len(ds), GRID_FRAMES, replace=False)
    rows = []
    for i in idx:
        item = ds[int(i)]
        rows.append(np.concatenate([to_uint8(item[k]) for k in (*RGB_KEYS, MASK_KEY)], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main():
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/gvla_libero")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    paths = load_paths()
    root = paths["lerobot_datasets"] / args.repo_id.split("/")[-1]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    deltas = {"action": [t / CONTROL_HZ for t in range(CHUNK_SIZE)]}
    ds = LeRobotDataset(args.repo_id, root=root, delta_timestamps=deltas, video_backend=args.video_backend)
    print(f"{ds.num_episodes} episodes, {ds.num_frames} frames, tasks: {list(ds.meta.tasks.index)}")
    print("splits:", {k: {s: len(v) for s, v in d.items()} for k, d in json.loads((root / 'splits.json').read_text()).items()})

    out_dir = paths["outputs"] / "dataset_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_grid(ds, out_dir / "random_frames.png")
    print(f"saved {out_dir / 'random_frames.png'} (agentview | wrist | mask)")

    item = ds[0]
    for key in (*RGB_KEYS, MASK_KEY, "observation.state", "observation.grounding", "action"):
        v = item[key]
        print(f"{key}: shape={tuple(v.shape)} min={v.min():.3f} max={v.max():.3f}")
    grounding = torch.stack([ds.hf_dataset[int(i)]["observation.grounding"] for i in np.linspace(0, len(ds) - 1, 500).astype(int)])
    print(f"normalized grounding coordinates: min={grounding[:, :15].min():.3f} max={grounding[:, :15].max():.3f}")

    spec = load_registry()["smolvla_libero"]
    ckpt = str(resolve_weights(spec["hub_id"], spec["folder"], paths))
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(ckpt, config=cfg).to(device).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                      preprocessor_overrides={"device_processor": {"device": device}})
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(0))
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i == args.batches:
                break
            batch = pre(batch)
            loss, _ = policy.forward(batch)
            losses.append(float(loss))
            print(f"batch {i}: action={tuple(batch['action'].shape)} "
                  f"tokens={tuple(batch['observation.language.tokens'].shape)} loss={losses[-1]:.4f}")
    print(f"smolvla_libero mean flow-matching loss on our dataset: {np.mean(losses):.4f}")


if __name__ == "__main__":
    main()
