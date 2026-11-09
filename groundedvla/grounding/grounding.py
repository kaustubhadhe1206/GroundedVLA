"""Mask -> point cloud -> robot-frame centroid -> 16-D grounding vector (Step A6).

Layout of the grounding vector:
    [0:3]   target centroid xyz
    [3:6]   destination centroid xyz
    [6:9]   end-effector xyz
    [9:12]  target - eef
    [12:15] destination - target
    [15]    target visibility flag
All positions are in the robot base frame. Depth is used only here, never as a model input.

Centroid = midpoint of the 5th and 95th percentile of the object's visible points per axis (a
robust bounding-box center), not the plain median the plan first specified. The camera sees one
side of an object (the inside of a bowl's far wall, the top of a box), which drags the median
~2.4 cm off a bowl's center; the robust box center stays within ~0.5 cm (checked Nov 1 against
the center of each object's collision geometry, see check_grounding.py).
"""
from dataclasses import dataclass, field

import numpy as np

from groundedvla.envs import camera_utils as cu

MIN_VISIBLE_PIXELS = 20
GROUNDING_DIM = 16
COORD_DIM = 15  # everything except the visibility flag
CENTROID_PERCENTILES = (5, 95)


@dataclass
class Normalizer:
    """Workspace center/half-extent from training data; saved with the dataset and checkpoint."""
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

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(np.asarray(d["center"], np.float32), np.asarray(d["half_extent"], np.float32))


def robust_center(points: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(points, CENTROID_PERCENTILES, axis=0)
    return (lo + hi) / 2


@dataclass
class CameraView:
    """One camera frame in OpenCV pixel order plus the camera model at that instant.

    Built from the *raw* (unflipped) obs arrays, and re-reads intrinsics/extrinsics every time,
    so camera perturbations (A7) are picked up automatically.
    """
    seg: np.ndarray          # (H, W) instance ids
    depth: np.ndarray        # (H, W) meters
    K: np.ndarray            # 3x3
    T_cam_to_base: np.ndarray  # 4x4

    @classmethod
    def from_obs(cls, sim, raw_obs: dict, camera: str = "agentview") -> "CameraView":
        seg = cu.opencv_order(raw_obs[f"{camera}_segmentation_instance"])[..., 0]
        depth = cu.opencv_order(cu.metric_depth(sim, raw_obs[f"{camera}_depth"]))[..., 0]
        h, w = seg.shape
        K = cu.intrinsics(sim, camera, h, w)
        T = cu.world_to_base(sim) @ cu.extrinsics(sim, camera)
        return cls(seg, depth, K, T)

    def points(self, inst_id: int) -> np.ndarray:
        """Robot-frame points of an object's visible pixels, (N, 3)."""
        pix = np.argwhere(self.seg == inst_id)
        return cu.transform(self.T_cam_to_base, cu.backproject(self.depth, pix, self.K))

    def centroid(self, inst_id: int) -> np.ndarray | None:
        """Robust robot-frame center of the object, or None if < MIN_VISIBLE_PIXELS are visible."""
        if np.count_nonzero(self.seg == inst_id) < MIN_VISIBLE_PIXELS:
            return None
        return robust_center(self.points(inst_id))


def object_centroid(sim, raw_obs: dict, inst_id: int, camera: str = "agentview") -> np.ndarray | None:
    return CameraView.from_obs(sim, raw_obs, camera).centroid(inst_id)


def eef_in_base(sim, raw_obs: dict) -> np.ndarray:
    return cu.transform(cu.world_to_base(sim), np.asarray(raw_obs["robot0_eef_pos"])[None])[0]


def compose(target: np.ndarray, dest: np.ndarray, eef: np.ndarray, visible: bool) -> np.ndarray:
    return np.concatenate([target, dest, eef, target - eef, dest - target, [float(visible)]]).astype(np.float32)


class GroundingTracker:
    """Per-episode state: keeps the last valid centroid when an object is occluded.

    Create a new tracker (or call reset) at the start of every episode.
    """

    def __init__(self, target_id: int, dest_id: int, normalizer: Normalizer | None = None,
                 camera: str = "agentview"):
        self.target_id, self.dest_id = target_id, dest_id
        self.normalizer, self.camera = normalizer, camera
        self.reset()

    def reset(self):
        self.last_target = np.zeros(3)
        self.last_dest = np.zeros(3)

    def __call__(self, sim, raw_obs: dict, normalize: bool = True) -> np.ndarray:
        view = CameraView.from_obs(sim, raw_obs, self.camera)
        t, d = view.centroid(self.target_id), view.centroid(self.dest_id)
        visible = t is not None
        if visible:
            self.last_target = t
        if d is not None:
            self.last_dest = d
        g = compose(self.last_target, self.last_dest, eef_in_base(sim, raw_obs), visible)
        return self.normalizer(g) if normalize and self.normalizer is not None else g
