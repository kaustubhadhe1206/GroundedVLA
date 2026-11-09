"""Evaluate one policy on tasks x conditions; one JSON line per episode, resumable (Steps A3, A4, B4).

Usage:
    python -m groundedvla.eval.run_eval --config configs/v1_rgb.yaml --policy replay --tasks T1 --episodes 10
    python -m groundedvla.eval.run_eval --config configs/v1_rgb.yaml --policy smolvla_ckpt --checkpoint <dir> \
        --eval-config configs/eval_ood.yaml
    # checkpoint selection / horizon tuning (B1): validation-demo initial states, never the test states
    python -m groundedvla.eval.run_eval --config configs/v1_rgb.yaml --checkpoint <run>/step_010000 \
        --split val --horizon 10 --tag step_010000_N10

Episodes already present in the output JSONL are skipped, so a disconnected Colab run just restarts
(plan Section 2.7).
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

import groundedvla.envs.compat  # noqa: F401
from groundedvla.envs.libero_wrapper import CameraPerturbation, LiberoWrapper
from groundedvla.eval.rollout import run_episode
from groundedvla.policies.chunking import ChunkExecutor
from groundedvla.policies.registry import build_policy, default_horizon, load_registry
from groundedvla.tasks import TASKS
from groundedvla.utils.paths import REPO_ROOT, load_paths


def val_init_states(manifest: dict, task_key: str, paths: dict) -> list[np.ndarray]:
    """Initial states of the task's validation demos (split manifest), from the raw LIBERO hdf5."""
    d = manifest["demos"][task_key]
    with h5py.File(Path(paths["libero_raw"]) / d["suite"] / f"{d['bddl_name']}_demo.hdf5") as f:
        return [f["data"][name]["states"][0] for name in d["val"]]


def done_episodes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["episode_idx"] for line in path.read_text().splitlines() if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-config", default=None, help="OOD conditions; default = in-distribution only")
    ap.add_argument("--policy", default="smolvla_ckpt", choices=list(load_registry()))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--mode", default=None, help="naive | temporal_ensemble (default: config eval.mode)")
    ap.add_argument("--horizon", type=int, default=None, help="naive execution horizon N (default: config)")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="test: LIBERO's fixed init states (+ OOD); val: validation-demo init states (selection)")
    ap.add_argument("--tag", default=None, help="sub-folder for this run, e.g. step_010000_N10")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--wandb", action="store_true", help="also log each episode to Weights & Biases")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = load_paths()
    manifest = json.loads((REPO_ROOT / cfg["data"]["manifest"]).read_text())
    conditions = {"in_distribution": {}} if args.split == "test" else {"val": {}}
    if args.eval_config and args.split == "test":
        conditions.update(yaml.safe_load(Path(args.eval_config).read_text())["conditions"])
    out_dir = Path(paths["outputs"]) / "eval" / cfg["name"] / args.policy
    if args.tag:
        out_dir = out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode or cfg["eval"].get("mode", "naive")
    # Our checkpoints use the config's horizon; pipeline-check models have their own (models.yaml).
    horizon = cfg["eval"]["execution_horizon"] if args.policy == "smolvla_ckpt" else default_horizon(args.policy)
    horizon = args.horizon or horizon
    base_seed = cfg["eval"]["seed"] if args.split == "test" else manifest["val_rollout_seed"]
    executor = ChunkExecutor(mode, horizon, cfg["eval"].get("te_m", 0.01))
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=cfg["logging"]["wandb_project"], name=f"eval_{cfg['name']}_{args.policy}",
                               job_type="eval", config={"policy": args.policy, "checkpoint": args.checkpoint,
                                                        "mode": mode, "horizon": horizon})

    for task_key in args.tasks or cfg["tasks"]:
        spec = TASKS[task_key]
        policy = build_policy(args.policy, spec, paths, checkpoint=args.checkpoint)
        for cond_name, cond in conditions.items():
            log = out_dir / f"{task_key}_{cond_name}.jsonl"
            finished = done_episodes(log)
            pert = CameraPerturbation(**cond["camera"]) if "camera" in cond else None
            # TODO(Week 3): position/scene OOD init states from split_manifest["ood"]
            env = LiberoWrapper(spec, with_depth_seg=cfg["model"]["variant"] != "v1", camera_perturbation=pert)
            init_states = env.init_states() if args.split == "test" else val_init_states(manifest, task_key, paths)
            default_n = cfg["eval"]["rollouts_per_task"] if args.split == "test" else manifest["val_rollouts_per_task"]
            n = args.episodes or cond.get("rollouts", default_n)
            for i in range(n):
                if i in finished:
                    continue
                init = policy.begin_episode(i)
                init = init_states[i % len(init_states)] if init is None else init
                video = None if args.no_video else out_dir / "videos" / f"{task_key}_{cond_name}_ep{i:03d}.mp4"
                r = run_episode(env, policy, executor, init_state=init, max_steps=spec.max_steps,
                                seed=base_seed + i, condition=cond_name, episode_idx=i, video_path=video)
                record = r.to_dict() | {"mode": mode, "horizon": horizon if mode == "naive" else None}
                with open(log, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if wandb_run is not None:
                    wandb_run.log(record)
                print(f"{task_key} {cond_name} ep{i}: success={r.success} steps={r.steps} "
                      f"grasp={r.grasp_success} collisions={r.collisions} err={r.final_pos_error_cm:.1f}cm "
                      f"failure={r.failure}")
            env.close()
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            print(f"{task_key} {cond_name}: success {np.mean([r['success'] for r in rows]):.2%} over {len(rows)} -> {log}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
