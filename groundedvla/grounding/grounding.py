"""Mask -> point cloud -> robot-frame centroid -> 16-D grounding vector (Step A6).

Layout of the grounding vector:
    [0:3]   target centroid xyz
    [3:6]   destination centroid xyz
    [6:9]   end-effector xyz
    [9:12]  target - eef
    [12:15] destination - target
    [15]    target visibility flag
All positions are in the robot base frame. Depth is used only here, never as a model input.
"""
from dataclasses import dataclass, field

import numpy as np

from groundedvla.envs import camera_utils as cu
from groundedvla.grounding.masks import binary_mask

MIN_VISIBLE_PIXELS = 20
GROUNDING_DIM = 16
COORD_DIM = 15  # everything except the visibility flag


@dataclass
class Normalizer:
    """Workspace center/half-extent from training data; saved with the checkpoint."""
    center: np.ndarray = field(default_factory=lambda: np.zeros(COORD_DIM, np.float32))
    half_extent: np.ndarray = field(default_factory=lambda: np.ones(COORD_DIM, np.float32))

    def __call__(self, g: np.ndarray) -> np.ndarray:
        out = g.copy()
        out[..., :COORD_DIM] = (g[..., :COORD_DIM] - self.center) / self.half_extent
        return out

    @classmethod
    def fit(cls, vectors: np.ndarray) -> "Normalizer":
        x = vectors[:, :COORD_DIM]
        lo, hi = x.min(0), x.max(0)
        return cls(((lo + hi) / 2).astype(np.float32), np.maximum((hi - lo) / 2, 1e-3).astype(np.float32))

    def state_dict(self):
        return {"center": self.center.tolist(), "half_extent": self.half_extent.tolist()}


def object_centroid(sim, raw_obs: dict, inst_id: int, camera="agentview", size=256):
    """Median robot-frame point of an object's visible pixels, or None if < MIN_VISIBLE_PIXELS.

    Uses the *unflipped* raw arrays so pixel coords match the MuJoCo camera model.
    """
    seg = raw_obs[f"{camera}_segmentation_instance"]
    mask = binary_mask(seg, inst_id)
    pix = np.argwhere(mask)
    if len(pix) < MIN_VISIBLE_PIXELS:
        return None
    depth = cu.metric_depth(sim, raw_obs[f"{camera}_depth"]).squeeze(-1)
    K = cu.intrinsics(sim, camera, size, size)
    pts_cam = cu.backproject(depth, pix, K)
    # TODO(A6): MuJoCo renders with OpenGL convention (row 0 = bottom). Verify the
    # row/col convention against check_grounding.py overlays before trusting this.
    pts_world = cu.transform(cu.extrinsics(sim, camera), pts_cam)
    pts_base = cu.transform(cu.world_to_base(sim), pts_world)
    return np.median(pts_base, axis=0)


class GroundingTracker:
    """Per-episode state: keeps the last valid centroid when the object is occluded."""

    def __init__(self, target_id: int, dest_id: int, normalizer: Normalizer | None = None):
        self.target_id, self.dest_id = target_id, dest_id
        self.normalizer = normalizer
        self.last_target = np.zeros(3)
        self.last_dest = np.zeros(3)

    def __call__(self, sim, raw_obs: dict, eef_base: np.ndarray) -> np.ndarray:
        t = object_centroid(sim, raw_obs, self.target_id)
        d = object_centroid(sim, raw_obs, self.dest_id)
        visible = t is not None
        if visible:
            self.last_target = t
        if d is not None:
            self.last_dest = d
        t, d = self.last_target, self.last_dest
        g = np.concatenate([t, d, eef_base, t - eef_base, d - t, [float(visible)]]).astype(np.float32)
        return self.normalizer(g) if self.normalizer is not None else g
