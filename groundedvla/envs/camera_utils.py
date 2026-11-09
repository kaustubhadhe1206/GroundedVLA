"""Camera intrinsics/extrinsics, depth conversion, and pixel conventions (plan Steps A5-A7).

Thin wrappers over robosuite.utils.camera_utils so the rest of the code never touches
MuJoCo conventions directly. Always re-query after a camera perturbation (A7).

Pixel conventions (checked Nov 1 by projecting ground-truth object positions onto the masks):
  raw obs arrays   OpenGL order: row 0 is the *bottom* of the image (robosuite IMAGE_CONVENTION)
  OpenCV order     raw[::-1] (vertical flip only). K and the extrinsics below assume this order.
  policy images    raw[::-1, ::-1] (LiberoWrapper.flip, the 180 deg flip LeRobot/OpenVLA use)
                   = OpenCV order mirrored left-right.
"""
import numpy as np

import groundedvla.envs.compat  # noqa: F401  (GL backend before robosuite is imported)
from robosuite.utils import camera_utils as rcu


def intrinsics(sim, camera_name: str, height: int, width: int) -> np.ndarray:
    """3x3 pinhole intrinsic matrix K (reads the current cam_fovy)."""
    return rcu.get_camera_intrinsic_matrix(sim, camera_name, height, width)


def extrinsics(sim, camera_name: str) -> np.ndarray:
    """4x4 camera-to-world pose (OpenCV convention: +z forward, +y down)."""
    return rcu.get_camera_extrinsic_matrix(sim, camera_name)


def metric_depth(sim, depth_normalized: np.ndarray) -> np.ndarray:
    """MuJoCo normalized depth buffer -> depth in meters."""
    return rcu.get_real_depth_map(sim, depth_normalized)


def opencv_order(raw_img: np.ndarray) -> np.ndarray:
    """Raw (OpenGL-order) obs array -> OpenCV pixel order, matching `intrinsics`/`extrinsics`."""
    return raw_img[::-1]


def opencv_to_policy_pixels(rows: np.ndarray, cols: np.ndarray, width: int):
    """OpenCV (row, col) -> (row, col) in the flipped images the policy and videos show."""
    return rows, width - 1 - cols


def world_to_base(sim, robot_base_body: str = "robot0_base") -> np.ndarray:
    """4x4 transform taking world-frame points into the robot base frame."""
    bid = sim.model.body_name2id(robot_base_body)
    T = np.eye(4)
    T[:3, :3] = sim.data.body_xmat[bid].reshape(3, 3)
    T[:3, 3] = sim.data.body_xpos[bid]
    return np.linalg.inv(T)


def backproject(depth_m: np.ndarray, pixels_vu: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels (N, 2) as (row, col) in OpenCV order + metric depth -> (N, 3) points in camera frame."""
    v, u = pixels_vu[:, 0], pixels_vu[:, 1]
    z = depth_m[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def project(points_world: np.ndarray, sim, camera_name: str, height: int, width: int) -> np.ndarray:
    """(N, 3) world points -> (N, 2) float (row, col) in OpenCV order (not clipped)."""
    T_wc = np.linalg.inv(extrinsics(sim, camera_name))
    pc = transform(T_wc, np.atleast_2d(points_world))
    uv = pc @ intrinsics(sim, camera_name, height, width).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv[:, ::-1]


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]
