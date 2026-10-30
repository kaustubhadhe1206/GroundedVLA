"""One closed-loop episode through the shared harness (Step A4).

Every policy in groundedvla/policies (expert replay, smolvla_base, smolvla_libero, OpenVLA,
our V1/V2/V3) runs through this one loop. A ChunkExecutor turns predicted chunks into one action
per step (naive chunking or temporal ensemble). scripts/run_sim.py uses the same loop, adding a
live viewer through `on_step`.
"""
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from groundedvla.envs.libero_wrapper import LiberoWrapper
from groundedvla.eval.metrics import EpisodeResult
from groundedvla.policies.base import Policy
from groundedvla.policies.chunking import ChunkExecutor
from groundedvla.tasks import CONTROL_HZ
from groundedvla.viz.viewer import VideoRecorder


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_episode(env: LiberoWrapper, policy: Policy, executor: ChunkExecutor, *, init_state,
                max_steps: int, seed: int, condition: str, episode_idx: int,
                video_path: Path | None = None,
                on_step: Callable[[LiberoWrapper, dict, int], bool] | None = None) -> EpisodeResult:
    """`on_step(env, obs, step)` runs after reset and after every step; returning False aborts."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    res = EpisodeResult(task=env.task.key, instruction=env.instruction, variant=policy.name,
                        checkpoint=policy.checkpoint, seed=seed, condition=condition, episode_idx=episode_idx)
    obs = env.reset(init_state, settle=policy.settle)
    policy.reset()
    executor.reset()
    recorder = VideoRecorder() if video_path is not None else None

    def timed_predict(o):
        _sync()
        t0 = time.perf_counter()
        chunk = policy.predict_chunk(o)
        _sync()
        res.latency_ms.append((time.perf_counter() - t0) * 1e3)
        return chunk

    aborted = on_step is not None and on_step(env, obs, 0) is False
    step = 0
    while not aborted and step < max_steps:
        action, _ = executor.act(obs, timed_predict)
        obs, _, _, info = env.step(env.action_from_policy(action))
        step += 1
        if recorder is not None:
            recorder.add(obs)
        if on_step is not None and on_step(env, obs, step) is False:
            aborted = True
        if info["success"]:
            res.success = True
            break

    res.steps, res.sim_time_s = step, step / CONTROL_HZ
    res.grounding_ms = list(getattr(policy, "grounding_ms", []))
    if not res.success:
        res.failure = "aborted" if aborted else "timeout"  # TODO(A4): failure_labels.label_failure
    if recorder is not None:
        saved = recorder.save(video_path)
        res.video = str(saved) if saved else None
    return res
