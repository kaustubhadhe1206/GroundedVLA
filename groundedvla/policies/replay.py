"""Expert demo replay (Step A2/A4 check): plays a LIBERO demo's recorded actions, one per step.

Episode i replays demo_i from its own recorded initial state, with no settle steps, because
the recorded actions start right at that state. A correct wrapper must reach 100% success.
"""
from pathlib import Path

import h5py
import numpy as np


class ReplayPolicy:
    name = "replay"
    settle = False

    def __init__(self, hdf5_path: str | Path):
        self.path = Path(hdf5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"demo file not found: {self.path} (run scripts/download_all.py --what data)")
        self.checkpoint = str(self.path)
        with h5py.File(self.path) as f:
            self.num_demos = len(f["data"])
        self.actions = np.zeros((0, 7), np.float32)
        self.t = 0

    def begin_episode(self, episode_idx: int) -> np.ndarray:
        with h5py.File(self.path) as f:
            demo = f["data"][f"demo_{episode_idx % self.num_demos}"]
            self.actions = demo["actions"][()].astype(np.float32)
            return demo["states"][0]

    def reset(self) -> None:
        self.t = 0

    def predict_chunk(self, obs: dict) -> np.ndarray:
        if self.t < len(self.actions):
            a = self.actions[self.t]
        else:  # demo exhausted without success: hold still until timeout
            a = np.array([0, 0, 0, 0, 0, 0, self.actions[-1, 6] if len(self.actions) else -1], np.float32)
        self.t += 1
        return a[None]
