"""Action-chunk execution modes (plan Section 1, Step A4).

naive              predict a chunk, execute its first N actions, predict again
temporal_ensemble  predict every step; execute the weighted average of every stored chunk that
                   covers the current step, w_i = exp(-m * i) with i = 0 the oldest chunk (ACT)
"""
from collections import deque
from typing import Callable

import numpy as np

MODES = ("naive", "temporal_ensemble")


class ChunkExecutor:
    def __init__(self, mode: str = "naive", horizon: int = 10, te_m: float = 0.01):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.mode, self.horizon, self.te_m = mode, horizon, te_m
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.queue: deque = deque()
        self.chunks: list[tuple[int, np.ndarray]] = []   # (start step, chunk) for temporal ensemble

    def act(self, obs: dict, predict: Callable[[dict], np.ndarray]) -> tuple[np.ndarray, bool]:
        """One action for this step, and whether the policy was queried."""
        predicted = False
        if self.mode == "naive":
            if not self.queue:
                self.queue.extend(np.asarray(predict(obs))[: self.horizon])
                predicted = True
            action = self.queue.popleft()
        else:
            self.chunks.append((self.t, np.asarray(predict(obs))))
            predicted = True
            self.chunks = [(s, c) for s, c in self.chunks if s + len(c) > self.t]
            preds = np.stack([c[self.t - s] for s, c in self.chunks])        # oldest first
            w = np.exp(-self.te_m * np.arange(len(preds)))
            action = (w[:, None] * preds).sum(0) / w.sum()
        self.t += 1
        return np.asarray(action, dtype=np.float32), predicted
