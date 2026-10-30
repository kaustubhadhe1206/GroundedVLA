"""Live MuJoCo viewer (laptop) and rollout video recording (Colab/CARC, or both).

LIBERO rebuilds the MuJoCo model on every reset (hard_reset=True, kept for fidelity with LIBERO
evaluation), so the viewer window is reopened whenever the model object changes, i.e. once
per episode. Closing the window stops the run.

Viewer keys (MuJoCo defaults): [ and ] cycle cameras, Esc returns to the free camera,
mouse drag rotates / pans, Tab and Shift+Tab toggle the side panels.
"""
import time
from pathlib import Path

import numpy as np

from groundedvla.tasks import CONTROL_HZ


class SimViewer:
    def __init__(self, realtime: bool = True, camera: str | None = "agentview"):
        self.realtime, self.camera = realtime, camera
        self.viewer, self._model = None, None
        self._last = 0.0
        self.closed_by_user = False

    def _open(self, sim):
        import mujoco
        import mujoco.viewer

        model, data = sim.model._model, sim.data._data
        self.viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        self._model = model
        if self.camera is not None:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
            if cid >= 0:
                with self.viewer.lock():
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.viewer.cam.fixedcamid = cid

    def sync(self, sim) -> bool:
        """Draw the current sim state. Returns False once the user has closed the window."""
        if self.viewer is not None and sim.model._model is self._model and not self.viewer.is_running():
            self.closed_by_user = True
            return False
        if self.viewer is None or sim.model._model is not self._model:
            self.close()
            self._open(sim)
        self.viewer.sync()
        if self.realtime:
            wait = 1.0 / CONTROL_HZ - (time.perf_counter() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.perf_counter()
        return True

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer, self._model = None, None


class VideoRecorder:
    """Side-by-side agentview | wrist frames (exactly what the policy sees) -> mp4 at 20 fps."""

    def __init__(self):
        self.frames: list[np.ndarray] = []

    def add(self, obs: dict):
        self.frames.append(np.concatenate([obs["observation.images.image"], obs["observation.images.image2"]], 1))

    def save(self, path: Path) -> Path | None:
        if not self.frames:
            return None
        import imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, self.frames, fps=CONTROL_HZ, macro_block_size=1)
        self.frames = []
        return path
