"""LIBERO environment wrapper: obs/action mapping, image flip, camera perturbation (A1, A2, A7).

Raw LIBERO images are upside down; RGB, depth, and segmentation are all flipped by 180
degrees *here and only here* so every downstream module sees upright images.

IMPORTANT: when flipping images, pixel coordinates no longer match the MuJoCo camera model.
grounding.py therefore works on the *unflipped* arrays kept in `obs["raw"]`.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import groundedvla.envs.compat  # noqa: F401  (GL backend + robosuite patch, before LIBERO)
from groundedvla.data.bddl_parser import TaskGoal, parse_bddl
from groundedvla.tasks import NUM_SETTLE_STEPS, TaskSpec

CAMERAS = ("agentview", "robot0_eye_in_hand")
IMG_SIZE = 256
# LIBERO convention: -1 = open, +1 = close. Demo actions use exactly {-1, +1} (checked Oct 30).
GRIPPER_OPEN, GRIPPER_CLOSE = -1.0, 1.0


@dataclass
class CameraPerturbation:
    """Agentview camera change applied at reset (A7, Section 6.3)."""
    yaw_deg: float = 0.0          # rotation around the workspace vertical axis
    height_m: float = 0.0
    fov_scale: float = 1.0
    pivot: tuple = (0.0, 0.0, 0.9)  # TODO(A7): set to workspace center in world frame


@dataclass
class LiberoWrapper:
    task: TaskSpec
    with_depth_seg: bool = True
    camera_perturbation: CameraPerturbation | None = None
    seed: int = 0
    goal: TaskGoal = field(init=False)

    def __post_init__(self):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bddl = Path(get_libero_path("bddl_files")) / self.task.suite / f"{self.task.bddl_name}.bddl"
        self.goal = parse_bddl(bddl)
        kwargs = dict(bddl_file_name=str(bddl), camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
                      camera_names=list(CAMERAS))
        if self.with_depth_seg:
            kwargs.update(camera_depths=True, camera_segmentations="instance")
        self.env = OffScreenRenderEnv(**kwargs)
        # Seed once at creation, like LIBERO/OpenVLA eval: it affects object placement even
        # when a fixed initial state is restored afterwards.
        self.env.seed(self.seed)

    @property
    def sim(self):
        return self.env.sim

    def _suite_and_index(self):
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[self.task.suite]()
        return suite, [t.name for t in suite.tasks].index(self.task.bddl_name)

    @property
    def instruction(self) -> str:
        """The benchmark's instruction, e.g. "pick up the black bowl ...". This is what the demos'
        `language_instruction` and every LIBERO-trained model use. The BDDL `(:language ...)` text
        differs ("Pick the akita black bowl ..."), so it is never fed to a policy."""
        if not hasattr(self, "_instruction"):
            suite, i = self._suite_and_index()
            self._instruction = suite.get_task(i).language
        return self._instruction

    def init_states(self) -> np.ndarray:
        """LIBERO's 50 fixed evaluation initial states for this task (in-distribution test)."""
        suite, i = self._suite_and_index()
        return suite.get_task_init_states(i)

    # ---------------- reset / step ----------------
    def reset(self, init_state: np.ndarray | None = None, settle: bool = True) -> dict:
        """settle=False for expert replay, whose recorded actions start at the init state."""
        raw = self.env.reset()
        if self.camera_perturbation is not None:
            self.apply_camera_perturbation(self.camera_perturbation)
        if init_state is not None:
            raw = self.env.set_init_state(init_state)
        for _ in range(NUM_SETTLE_STEPS if settle else 0):
            raw, *_ = self.env.step(self.dummy_action())
        return self.format_obs(raw)

    def step(self, action: np.ndarray):
        raw, reward, done, info = self.env.step(action)
        info["success"] = bool(self.env.check_success())
        return self.format_obs(raw), reward, done, info

    @staticmethod
    def dummy_action() -> np.ndarray:
        return np.array([0, 0, 0, 0, 0, 0, GRIPPER_OPEN], dtype=np.float32)

    # ---------------- observation mapping ----------------
    def format_obs(self, raw: dict) -> dict:
        """LIBERO raw obs -> SmolVLA-style dict (keys documented in README)."""
        obs = {
            "observation.images.image": flip(raw["agentview_image"]),
            "observation.images.image2": flip(raw["robot0_eye_in_hand_image"]),
            "observation.state": robot_state(raw),
            "task": self.instruction,
            "raw": raw,  # unflipped arrays for grounding / logging
        }
        if self.with_depth_seg:
            obs["depth.agentview"] = flip(raw["agentview_depth"])
            obs["seg.agentview"] = flip(raw["agentview_segmentation_instance"])
        return obs

    def action_from_policy(self, a: np.ndarray) -> np.ndarray:
        """SmolVLA 7-D output -> LIBERO action. TODO(A2): gripper binarization/sign check."""
        a = np.asarray(a, dtype=np.float32)[:7].copy()
        a[6] = GRIPPER_CLOSE if a[6] > 0 else GRIPPER_OPEN
        return np.clip(a, -1.0, 1.0)

    # ---------------- camera perturbation (A7) ----------------
    def apply_camera_perturbation(self, p: CameraPerturbation):
        """Rotate agentview around a vertical axis through `pivot`, raise it, scale FOV.

        TODO(A7): confirm this survives env.reset() (LIBERO may rebuild the model XML).
        """
        m = self.sim.model
        cid = m.camera_name2id("agentview")
        pos = m.cam_pos[cid].copy()
        quat_wxyz = m.cam_quat[cid].copy()
        R_yaw = Rotation.from_euler("z", p.yaw_deg, degrees=True)
        pivot = np.asarray(p.pivot)
        m.cam_pos[cid] = R_yaw.apply(pos - pivot) + pivot + np.array([0, 0, p.height_m])
        q = R_yaw * Rotation.from_quat(np.roll(quat_wxyz, -1))
        m.cam_quat[cid] = np.roll(q.as_quat(), 1)
        m.cam_fovy[cid] = m.cam_fovy[cid] * p.fov_scale
        self.sim.forward()

    def close(self):
        self.env.close()


def flip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ::-1])


def quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """robosuite's quat2axisangle (also used by LeRobot's LiberoProcessorStep and OpenVLA).

    Not scipy's as_rotvec: the LIBERO gripper points down, i.e. a rotation of ~180 deg, where
    the two disagree (scipy keeps the angle in [0, pi], this formula uses 2*acos(w) in [0, 2*pi]),
    which would flip the sign of the state vector relative to the pretrained checkpoints.
    """
    w = float(np.clip(quat_xyzw[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return quat_xyzw[:3] * 2.0 * np.arccos(w) / den


def robot_state(raw: dict) -> np.ndarray:
    """8-D state: eef pos (3), eef axis-angle (3), gripper finger qpos (2)."""
    axis_angle = quat2axisangle(np.asarray(raw["robot0_eef_quat"], dtype=np.float64))
    return np.concatenate([raw["robot0_eef_pos"], axis_angle, raw["robot0_gripper_qpos"]]).astype(np.float32)
