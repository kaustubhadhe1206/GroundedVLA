"""Fine-tune SmolVLA for V1 / V2 / V3 with one shared loop (Steps B1, B3).

Starts from lerobot/smolvla_base with our input features (2 RGB cameras, + mask for V2/V3), the 8-D
state and the 7-D action; smolvla_base pads state and action to 32-D internally, so its weights load
unchanged. Backbone frozen (freeze_vision_encoder + train_expert_only, LeRobot defaults). The optimizer,
grad clipping and cosine schedule match LeRobot's SmolVLA presets, with our lr / warmup / steps.

Checkpoints (plan Section 2.7), written to <checkpoints>/<run>/step_XXXXXX/:
  config.json, model.safetensors, policy_{pre,post}processor*   -> loadable by run_sim / run_eval
  training_state.pt   optimizer, scheduler, RNG states, step, W&B id, commit (latest 2 only)
  grounding_norm.json, train_config.yaml
Each one is written to a temporary folder and renamed, so a disconnect can't leave a half-written
checkpoint. Kept: the 2 newest, the selection steps (10k/15k/20k) and the lowest-val-loss one.
Starting again with the same config resumes from the newest checkpoint automatically.

Usage:
    python -m groundedvla.train.train --config configs/v1_rgb.yaml
    python -m groundedvla.train.train --config configs/v1_rgb.yaml --max-steps 200 --run-name v1_throughput --no-save
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from groundedvla.policies.registry import resolve_weights
from groundedvla.tasks import CONTROL_HZ
from groundedvla.utils.paths import REPO_ROOT, load_paths

TRAINING_STATE = "training_state.pt"
RUN_STATE = "run_state.json"
KEEP_LATEST = 2


# ----------------------------------------------------------------------------- data
def dataset_root(repo_id: str, paths: dict) -> Path:
    """Fast local copy first (Colab /content/local), then the persistent one."""
    name = repo_id.split("/")[-1]
    for p in (Path(paths["local_datasets"]) / "lerobot" / name, Path(paths["lerobot_datasets"]) / name):
        if (p / "meta" / "info.json").exists():
            return p
    raise FileNotFoundError(f"dataset {name} not found under {paths['local_datasets']}/lerobot or {paths['lerobot_datasets']}")


def split_episodes(root: Path, tasks: list[str]) -> tuple[list[int], list[int]]:
    splits = json.loads((root / "splits.json").read_text())
    train = sorted(i for t in tasks for i in splits[t]["train"])
    val = sorted(i for t in tasks for i in splits[t]["val"])
    assert not set(train) & set(val), "train/val overlap"
    return train, val


def make_datasets(cfg: dict, root: Path, video_backend: str | None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    train_ids, val_ids = split_episodes(root, cfg["tasks"])
    deltas = {"action": [t / CONTROL_HZ for t in range(cfg["model"]["chunk_size"])]}
    kw = dict(root=root, delta_timestamps=deltas, video_backend=video_backend)
    train = LeRobotDataset(cfg["data"]["repo_id"], episodes=train_ids, **kw)
    val = LeRobotDataset(cfg["data"]["repo_id"], episodes=val_ids, **kw)
    return train, val


def infinite_loader(ds, batch_size: int, num_workers: int, seed: int, start_epoch: int = 0):
    epoch = start_epoch
    while True:
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=False,
            prefetch_factor=4 if num_workers else None, generator=torch.Generator().manual_seed(seed + epoch))
        yield from loader
        epoch += 1


# ----------------------------------------------------------------------------- policy
def make_policy_config(cfg: dict, base: Path, ds_meta, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.utils import dataset_to_policy_features

    m, t = cfg["model"], cfg["train"]
    feats = dataset_to_policy_features(ds_meta.features)
    pc = PreTrainedConfig.from_pretrained(str(base))
    pc.input_features = {k: feats[k] for k in [*m["image_keys"], "observation.state"]}
    pc.output_features = {"action": feats["action"]}
    pc.chunk_size = pc.n_action_steps = m["chunk_size"]
    pc.freeze_vision_encoder = m["freeze_vision_encoder"]
    pc.train_expert_only = m["train_expert_only"]
    pc.optimizer_lr = t["lr"]
    pc.optimizer_weight_decay = t["weight_decay"]
    pc.scheduler_warmup_steps = t["warmup_steps"]
    pc.scheduler_decay_steps = t["steps"]          # full cosine over the run, no auto-rescaling
    pc.device = device
    return pc


def policy_class(variant: str):
    if variant in ("v1", "v2"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    raise NotImplementedError("V3 training is wired up in Step B2 (grounded_smolvla.py)")


# ----------------------------------------------------------------------------- augmentation
class ImageAugmentation:
    """Per-sample camera-style augmentation for the V1+aug baseline (plan Step B5 / improvement 4).

    Random affine (rotation, shift, scale, shear) + perspective warp approximate a moved camera;
    color jitter covers lighting. Applied only to `keys` (the agentview by default, the camera our
    viewpoint OOD conditions move), on the GPU after the preprocessor, training batches only.
    """

    def __init__(self, keys: list[str], spec: dict):
        from torchvision.transforms import v2

        a, ops = spec.get("affine"), []
        if a:
            ops.append(v2.RandomAffine(degrees=a.get("degrees", 0), translate=(a.get("translate", 0),) * 2,
                                       scale=tuple(a.get("scale", (1, 1))), shear=a.get("shear", 0)))
        if spec.get("perspective"):
            ops.append(v2.RandomPerspective(distortion_scale=spec["perspective"], p=spec.get("perspective_p", 0.5)))
        if spec.get("color_jitter"):
            c = spec["color_jitter"]
            ops.append(v2.ColorJitter(c.get("brightness", 0), c.get("contrast", 0), c.get("saturation", 0), c.get("hue", 0)))
        self.keys, self.prob = keys, spec.get("prob", 1.0)
        self.transform = v2.Compose(ops)

    def __call__(self, batch: dict) -> dict:
        for k in self.keys:
            imgs = batch[k]
            out = imgs.clone()
            for i in range(imgs.shape[0]):                 # per-sample random parameters
                if torch.rand(()) < self.prob:
                    out[i] = self.transform(imgs[i]).clamp_(0, 1)
            batch[k] = out
        return batch


def make_augmentation(cfg: dict) -> ImageAugmentation | None:
    spec = cfg["train"].get("augment")
    if not spec:
        return None
    keys = spec.get("keys", ["observation.images.image"])
    if "observation.images.mask" in cfg["model"]["image_keys"]:
        # the mask would need the same geometric transform as the agentview; not supported
        raise ValueError("image augmentation is only for the RGB-only baseline (no mask input)")
    return ImageAugmentation(keys, spec)


# ----------------------------------------------------------------------------- checkpoints
def list_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))


def latest_resumable(run_dir: Path) -> Path | None:
    ckpts = [p for p in list_checkpoints(run_dir) if (p / TRAINING_STATE).exists()]
    return ckpts[-1] if ckpts else None


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(run_dir: Path, step: int, policy, pre, post, train_state: dict, cfg: dict, ds_root: Path) -> Path:
    final, tmp = run_dir / f"step_{step:06d}", run_dir / f".tmp_step_{step:06d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    policy.save_pretrained(tmp)
    pre.save_pretrained(tmp)
    post.save_pretrained(tmp)
    torch.save(train_state, tmp / TRAINING_STATE)
    (tmp / "train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if (ds_root / "grounding_norm.json").exists():
        shutil.copy2(ds_root / "grounding_norm.json", tmp / "grounding_norm.json")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final


def prune_checkpoints(run_dir: Path, keep_steps: set[int], best_step: int | None) -> None:
    """Keep the newest KEEP_LATEST in full; keep selection/best steps as weights only; delete the rest."""
    ckpts = list_checkpoints(run_dir)
    latest = set(ckpts[-KEEP_LATEST:])
    for p in ckpts:
        step = int(p.name.split("_")[1])
        if p in latest:
            continue
        if step in keep_steps or step == best_step:
            (p / TRAINING_STATE).unlink(missing_ok=True)   # optimizer state only needed to resume
        else:
            shutil.rmtree(p)


# ----------------------------------------------------------------------------- loop
@torch.no_grad()
def validation_loss(policy, pre, val_ds, batch_size: int, max_batches: int, num_workers: int, use_bf16: bool) -> float:
    """Deterministic flow-matching loss on the val demos (fixed noise/time seed, own RNG)."""
    policy.eval()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    losses = []
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(12345)
        for i, batch in enumerate(loader):
            if i == max_batches:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss, _ = policy.forward(pre(batch))
            losses.append(float(loss))
    policy.train()
    return float(np.mean(losses))


def main():
    from lerobot.policies.factory import make_pre_post_processors

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (default: config); a non-default seed trains run <name>_s<seed>")
    ap.add_argument("--run-name", default=None, help="default: config name (+ _s<seed>)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (throughput test); schedule unchanged")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--video-backend", default=None, help="torchcodec | pyav (default: LeRobot's choice)")
    ap.add_argument("--no-save", action="store_true", help="never write checkpoints (throughput test)")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints of this run")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    t = cfg["train"]
    default_seed = t["seed"]
    if args.seed is not None:
        t["seed"] = args.seed               # saved into every checkpoint's train_config.yaml
    paths = load_paths()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = t.get("precision") == "bf16" and device == "cuda" and torch.cuda.is_bf16_supported()
    batch_size = args.batch_size or t["batch_size"]
    num_workers = t.get("num_workers", 8) if args.num_workers is None else args.num_workers
    total_steps = t["steps"]
    stop_step = min(total_steps, args.max_steps or total_steps)
    run_name = args.run_name or (cfg["name"] if t["seed"] == default_seed else f"{cfg['name']}_s{t['seed']}")
    run_dir = Path(paths["checkpoints"]) / run_name
    augment = make_augmentation(cfg)
    if augment is not None:
        print(f"[train] augmenting {augment.keys} per sample: {augment.transform}")
    run_dir.mkdir(parents=True, exist_ok=True)

    from groundedvla.utils.colab import commit_hash

    commit = commit_hash()
    ds_root = dataset_root(cfg["data"]["repo_id"], paths)
    train_ds, val_ds = make_datasets(cfg, ds_root, args.video_backend)
    print(f"[train] run={run_name} commit={commit} device={device} bf16={use_bf16} batch={batch_size}")
    print(f"[train] dataset {ds_root}: train {train_ds.num_episodes} episodes / {train_ds.num_frames} frames, "
          f"val {val_ds.num_episodes} / {val_ds.num_frames}")

    base = resolve_weights("lerobot/smolvla_base", "smolvla_base", paths)
    resume_from = None if args.no_resume else latest_resumable(run_dir)
    pc = make_policy_config(cfg, base, train_ds.meta, device)
    cls = policy_class(cfg["model"]["variant"])
    policy = cls.from_pretrained(str(resume_from or base), config=pc).to(device)
    policy.train()
    pre, post = make_pre_post_processors(pc, dataset_stats=train_ds.meta.stats)

    params = [p for p in policy.parameters() if p.requires_grad]
    n_train, n_all = sum(p.numel() for p in params), sum(p.numel() for p in policy.parameters())
    print(f"[train] trainable params {n_train / 1e6:.1f}M of {n_all / 1e6:.1f}M")
    opt_cfg = pc.get_optimizer_preset()
    optimizer = torch.optim.AdamW(params, lr=opt_cfg.lr, betas=opt_cfg.betas, eps=opt_cfg.eps,
                                  weight_decay=opt_cfg.weight_decay)
    scheduler = pc.get_scheduler_preset().build(optimizer, total_steps)

    step, wandb_id, best = 0, None, {"val_loss": float("inf"), "step": None}
    run_state_path = run_dir / RUN_STATE
    if resume_from is not None:
        st = torch.load(resume_from / TRAINING_STATE, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        set_rng_state(st["rng"])
        step, wandb_id = st["step"], st.get("wandb_id")
        print(f"[train] resumed from {resume_from} at step {step}")
    else:
        random.seed(t["seed"])
        np.random.seed(t["seed"])
        torch.manual_seed(t["seed"])
    if run_state_path.exists() and resume_from is not None:
        best = json.loads(run_state_path.read_text()).get("best", best)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=cfg["logging"]["wandb_project"], name=run_name, id=wandb_id,
                               resume="allow", config=cfg | {"commit": commit})
        wandb_id = wandb_run.id

    log_every, val_every = t.get("log_every", 100), t.get("val_every", 1000)
    ckpt_every, keep_steps = t["checkpoint_every"], set(t.get("select_from_steps", []))
    metrics_path = run_dir / "metrics.jsonl"
    grad_clip = opt_cfg.grad_clip_norm
    batches = infinite_loader(train_ds, batch_size, num_workers, t["seed"], start_epoch=step)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def log(rec: dict) -> None:
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if wandb_run is not None:
            wandb_run.log(rec, step=rec["step"])

    t_last, loss_acc, gn_acc, n_acc, data_s = time.perf_counter(), 0.0, 0.0, 0, 0.0
    while step < stop_step:
        t_data = time.perf_counter()
        batch = next(batches)
        data_s += time.perf_counter() - t_data
        batch = pre(batch)
        if augment is not None:
            batch = augment(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss, _ = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        loss_acc, gn_acc, n_acc = loss_acc + loss.item(), gn_acc + grad_norm.item(), n_acc + 1

        if step % log_every == 0 or step == stop_step:
            now = time.perf_counter()
            rec = {"step": step, "train_loss": loss_acc / n_acc, "grad_norm": gn_acc / n_acc,
                   "lr": optimizer.param_groups[0]["lr"], "steps_per_s": n_acc / (now - t_last),
                   "data_frac": data_s / (now - t_last)}
            if device == "cuda":
                rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            print(f"[train] step {step:6d} loss {rec['train_loss']:.4f} gnorm {rec['grad_norm']:.2f} "
                  f"lr {rec['lr']:.2e} {rec['steps_per_s']:.2f} it/s (data {rec['data_frac']:.0%})"
                  + (f" mem {rec['gpu_mem_gb']:.1f}GB" if "gpu_mem_gb" in rec else ""), flush=True)
            log(rec)
            t_last, loss_acc, gn_acc, n_acc, data_s = now, 0.0, 0.0, 0, 0.0

        if step % val_every == 0 or step == stop_step:
            vl = validation_loss(policy, pre, val_ds, batch_size, t.get("val_batches", 25), num_workers, use_bf16)
            print(f"[train] step {step:6d} val_loss {vl:.4f}", flush=True)
            log({"step": step, "val_loss": vl})
            if vl < best["val_loss"] and step % ckpt_every == 0:
                best = {"val_loss": vl, "step": step}
            t_last = time.perf_counter()

        if not args.no_save and (step % ckpt_every == 0 or step == total_steps):
            state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(),
                     "step": step, "wandb_id": wandb_id, "commit": commit, "config": cfg}
            path = save_checkpoint(run_dir, step, policy, pre, post, state, cfg, ds_root)
            run_state_path.write_text(json.dumps({"best": best, "last_step": step, "commit": commit}, indent=2))
            prune_checkpoints(run_dir, keep_steps, best["step"])
            print(f"[train] saved {path} (best val step so far: {best['step']})", flush=True)
            t_last = time.perf_counter()

    if wandb_run is not None:
        wandb_run.finish()
    print(f"[train] done at step {step}; checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
