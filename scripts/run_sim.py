"""Run one model in LIBERO, episode after episode, in a live MuJoCo window or headless with videos.

Models (configs/models.yaml): replay, smolvla_base, smolvla_libero, smolvla_ckpt, openvla_7b

Laptop (opens the MuJoCo viewer):
    .\\run replay --task T1 --episodes 3
    .\\run smolvla_libero --task T2 --episodes 5
    .\\run smolvla_base --task T1 --mode temporal_ensemble
    .\\run smolvla_ckpt --checkpoint D:/groundedvla/checkpoints/v1_rgb/best --task T3
Colab / CARC (no display):
    python scripts/run_sim.py openvla_7b --task T1 --episodes 10 --viewer none --save-video

Tasks: T1..T4 (plan Section 3.1) or any LIBERO task as <suite>:<task_id>, e.g. libero_spatial:4.
Results: one JSON line per episode in <outputs>/run_sim/<model>/<task>.jsonl (+ mp4 with --save-video).
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import groundedvla.envs.compat  # noqa: E402,F401  (must precede LIBERO / robosuite imports)
from groundedvla.envs.libero_wrapper import CameraPerturbation, LiberoWrapper  # noqa: E402
from groundedvla.eval.metrics import wilson_ci  # noqa: E402
from groundedvla.eval.rollout import run_episode  # noqa: E402
from groundedvla.policies.chunking import MODES, ChunkExecutor  # noqa: E402
from groundedvla.policies.registry import build_policy, default_horizon, load_registry  # noqa: E402
from groundedvla.tasks import resolve_task  # noqa: E402
from groundedvla.utils.paths import load_paths  # noqa: E402


def has_display() -> bool:
    return platform.system() in ("Windows", "Darwin") or bool(os.environ.get("DISPLAY"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=list(load_registry()), help="model name from configs/models.yaml")
    ap.add_argument("--task", default="T1", help="T1..T7 or <suite>:<task_id> (default T1)")
    ap.add_argument("--episodes", type=int, default=3, help="number of episodes (default 3)")
    ap.add_argument("--start", type=int, default=0, help="first initial-state / demo index (default 0)")
    ap.add_argument("--checkpoint", default=None, help="model dir (required for smolvla_ckpt)")
    ap.add_argument("--mode", choices=MODES, default="naive", help="action chunk execution mode")
    ap.add_argument("--horizon", type=int, default=None, help="N actions per chunk in naive mode")
    ap.add_argument("--te-m", type=float, default=0.01, help="temporal ensemble weight decay m")
    ap.add_argument("--max-steps", type=int, default=None, help="override the task's episode limit")
    ap.add_argument("--viewer", choices=["window", "none"], default=None,
                    help="live MuJoCo window (default when a display exists) or headless")
    ap.add_argument("--fast", action="store_true", help="viewer: don't slow down to real time (20 Hz)")
    ap.add_argument("--save-video", action="store_true", help="save agentview|wrist mp4 per episode")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--load-in-4bit", action="store_true", help="OpenVLA on small GPUs (needs bitsandbytes)")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera-yaw", type=float, default=0.0, help="rotate agentview around the workspace (degrees)")
    ap.add_argument("--camera-height", type=float, default=0.0, help="raise agentview (meters)")
    ap.add_argument("--camera-fov-scale", type=float, default=1.0, help="scale agentview field of view")
    return ap.parse_args()


def main():
    args = parse_args()
    paths = load_paths()
    task = resolve_task(args.task)
    viewer_on = (args.viewer or ("window" if has_display() else "none")) == "window"
    horizon = args.horizon or default_horizon(args.model)
    max_steps = args.max_steps or task.max_steps

    print(f"[run_sim] model={args.model} task={task.key} ({task.suite}) episodes={args.episodes} "
          f"mode={args.mode}{f' N={horizon}' if args.mode == 'naive' else f' m={args.te_m}'} "
          f"max_steps={max_steps} viewer={'window' if viewer_on else 'none'} profile={paths['profile']}")
    t0 = time.perf_counter()
    policy = build_policy(args.model, task, paths, checkpoint=args.checkpoint, device=args.device,
                          load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit)
    print(f"[run_sim] loaded {policy.checkpoint} in {time.perf_counter() - t0:.1f}s")

    perturbed = args.camera_yaw or args.camera_height or args.camera_fov_scale != 1.0
    camera = CameraPerturbation(args.camera_yaw, args.camera_height, args.camera_fov_scale) if perturbed else None
    env = LiberoWrapper(task, with_depth_seg=False, seed=args.seed, camera_perturbation=camera)
    print(f"[run_sim] instruction: {env.instruction!r}")
    init_states = env.init_states()
    executor = ChunkExecutor(args.mode, horizon, args.te_m)
    out_dir = Path(paths["outputs"]) / "run_sim" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    # Perturbed runs get their own log and video names, e.g. T3_yaw20, so they never mix with or
    # overwrite the unperturbed T3 results.
    condition = "run_sim"
    if perturbed:
        parts = [f"yaw{args.camera_yaw:g}" if args.camera_yaw else "",
                 f"h{args.camera_height:+g}" if args.camera_height else "",
                 f"fov{args.camera_fov_scale:g}" if args.camera_fov_scale != 1.0 else ""]
        condition = "camera_" + "_".join(p for p in parts if p)
    stem = task.key.replace(":", "_") + ("" if condition == "run_sim" else "_" + condition.removeprefix("camera_"))
    log_path = out_dir / f"{stem}.jsonl"

    viewer = None
    if viewer_on:
        from groundedvla.viz.viewer import SimViewer

        viewer = SimViewer(realtime=not args.fast)

    def on_step(e, _obs, _step):
        return viewer.sync(e.sim)

    successes, done = 0, 0
    try:
        for ep in range(args.start, args.start + args.episodes):
            init = policy.begin_episode(ep)
            if init is None:
                init = init_states[ep % len(init_states)]
            video = out_dir / "videos" / f"{stem}_ep{ep:03d}.mp4" if args.save_video else None
            r = run_episode(env, policy, executor, init_state=init, max_steps=max_steps, seed=args.seed + ep,
                            condition=condition, episode_idx=ep, video_path=video,
                            on_step=on_step if viewer is not None else None)
            if r.failure == "aborted":
                print("[run_sim] viewer closed, stopping")
                break
            done += 1
            successes += r.success
            d = r.to_dict() | {"mode": args.mode, "horizon": horizon if args.mode == "naive" else None}
            with open(log_path, "a") as f:
                f.write(json.dumps(d) + "\n")
            lat = d["latency_ms_median"]
            msg = f"[run_sim] ep {ep:3d}: {'SUCCESS' if r.success else 'fail   '} steps={r.steps:3d}"
            msg += f" policy_call={lat:.0f}ms" if lat is not None else ""
            msg += f" video={r.video}" if r.video else ""
            print(msg)
    except KeyboardInterrupt:
        print("\n[run_sim] interrupted")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    if done:
        lo, hi = wilson_ci(successes, done)
        print(f"[run_sim] success {successes}/{done} = {successes / done:.0%} (95% CI {lo:.0%}-{hi:.0%}) "
              f"| log: {log_path}")


if __name__ == "__main__":
    main()
