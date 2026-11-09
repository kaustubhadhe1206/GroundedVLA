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
    """Agentview camera change applied at reset (A7, Section 6.3).

    The yaw rotates the camera around a vertical axis through `pivot`, so it orbits the
    workspace and keeps looking at it. pivot=None uses the point where the unperturbed optical
    axis meets the table top (world frame).
    """
    yaw_deg: float = 0.0
    height_m: float = 0.0
    fov_scale: float = 1.0
    pivot: tuple | None = None


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
            raw = self._render_obs()  # reset() rendered with the original camera
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
        """Policy 7-D output -> LIBERO action: clip to [-1, 1], binarize the gripper (+1 = close)."""
        a = np.asarray(a, dtype=np.float32)[:7].copy()
        a[6] = GRIPPER_CLOSE if a[6] > 0 else GRIPPER_OPEN
        return np.clip(a, -1.0, 1.0)

    def _render_obs(self) -> dict:
        self.env.env._update_observables(force=True)
        return self.env.env._get_observations()

    # ---------------- scene queries (A4 metrics, A6 ground truth) ----------------
    @property
    def robosuite_env(self):
        return self.env.env

    def scene_objects(self) -> dict[str, str]:
        """{BDDL name: root body name} for movable objects and fixtures."""
        e = self.robosuite_env
        return {n: o.root_body for d in (e.objects_dict, e.fixtures_dict) for n, o in d.items()}

    def movable_objects(self) -> list[str]:
        return list(self.robosuite_env.objects_dict)

    def object_center(self, name: str) -> np.ndarray:
        """World-frame center of an object's collision geometry (axis-aligned box).

        Ground truth for the grounding check. Not the body origin: e.g. the bowl's origin is at
        its bottom, ~3 cm below its center.
        """
        geoms = object_geoms(self.sim, self.scene_objects()[name])
        return geoms_aabb_center(self.sim, geoms)

    def eval_goal(self) -> bool:
        """The BDDL goal predicate, evaluated now (same check as LIBERO's success)."""
        e = self.robosuite_env
        return all(e._eval_predicate(s) for s in e.parsed_problem["goal_state"])

    # ---------------- camera perturbation (A7) ----------------
    def apply_camera_perturbation(self, p: CameraPerturbation):
        """Rotate agentview around a vertical axis through the pivot, raise it, scale FOV.

        LIBERO rebuilds the model on every reset (hard_reset=True), which restores the original
        camera, so this runs after each env.reset(). If the model was *not* rebuilt, the saved
        original pose is restored first, so perturbations never accumulate.
        """
        m = self.sim.model
        cid = m.camera_name2id("agentview")
        if getattr(self, "_cam0", None) is not None and self._cam0[0] is m._model:
            m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = self._cam0[1:]
            self.sim.forward()
        self._cam0 = (m._model, m.cam_pos[cid].copy(), m.cam_quat[cid].copy(), float(m.cam_fovy[cid]))
        assert m.cam_bodyid[cid] == 0, "agentview must be attached to the world body"

        pivot = np.asarray(p.pivot) if p.pivot is not None else self.camera_look_at("agentview")
        pos, quat_wxyz = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
        R_yaw = Rotation.from_euler("z", p.yaw_deg, degrees=True)
        m.cam_pos[cid] = R_yaw.apply(pos - pivot) + pivot + np.array([0, 0, p.height_m])
        q = R_yaw * Rotation.from_quat(np.roll(quat_wxyz, -1))
        m.cam_quat[cid] = np.roll(q.as_quat(), 1)
        m.cam_fovy[cid] = m.cam_fovy[cid] * p.fov_scale
        self.sim.forward()

    def camera_look_at(self, camera: str) -> np.ndarray:
        """Where the camera's optical axis (MuJoCo cameras look along their -z) meets the table top."""
        cid = self.sim.model.camera_name2id(camera)
        origin = self.sim.data.cam_xpos[cid].copy()
        direction = -self.sim.data.cam_xmat[cid].reshape(3, 3)[:, 2]
        t = (table_top_z(self.sim) - origin[2]) / direction[2]
        return origin + t * direction

    def close(self):
        self.env.close()


def flip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ::-1])


def body_descendants(model, body_id: int) -> np.ndarray:
    """Ids of `body_id` and all bodies below it (MuJoCo lists parents before children)."""
    keep = np.zeros(model.nbody, bool)
    keep[body_id] = True
    for b in range(body_id + 1, model.nbody):
        keep[b] = keep[model.body_parentid[b]]
    return np.flatnonzero(keep)


def object_geoms(sim, root_body: str, collision_only: bool = True) -> np.ndarray:
    m = sim.model._model
    bodies = body_descendants(m, sim.model.body_name2id(root_body))
    g = np.flatnonzero(np.isin(m.geom_bodyid, bodies))
    if collision_only:
        g = g[(m.geom_contype[g] != 0) | (m.geom_conaffinity[g] != 0)]
    return g


def geoms_aabb_center(sim, geom_ids: np.ndarray) -> np.ndarray:
    m, d = sim.model._model, sim.data._data
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in geom_ids:
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        ext = np.abs(R) @ m.geom_aabb[g, 3:]
        lo, hi = np.minimum(lo, c - ext), np.maximum(hi, c + ext)
    return (lo + hi) / 2


def table_top_z(sim) -> float:
    g = sim.model.geom_name2id("table_collision")
    return float(sim.data.geom_xpos[g][2] + sim.model.geom_size[g][2])


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
