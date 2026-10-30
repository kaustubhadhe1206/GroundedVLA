"""Common policy interface used by the rollout harness and scripts/run_sim.py."""
from typing import Protocol

import numpy as np


class Policy(Protocol):
    name: str          # registry name, e.g. "smolvla_libero"
    checkpoint: str    # path or Hub id, logged with every episode
    settle: bool       # run the 10 no-op settle steps after reset (False for expert replay)

    def begin_episode(self, episode_idx: int) -> np.ndarray | None:
        """Called before each episode. Return an initial state to override LIBERO's, else None."""
        ...

    def reset(self) -> None:
        """Clear any internal state (action queues, caches) at the start of an episode."""
        ...

    def predict_chunk(self, obs: dict) -> np.ndarray:
        """(T, 7) actions in LIBERO's action space for the current observation. T >= 1."""
        ...
