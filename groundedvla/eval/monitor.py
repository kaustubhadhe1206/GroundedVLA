"""Per-episode metric signals read from the simulator (Step A4; definitions in plan Part A).

Create one EpisodeMonitor after env.reset(): LIBERO rebuilds the model on every reset, and the
starting object poses are the reference for "lifted" and "displaced". Call update(obs) after every
control step and finish(result) once at the end.
"""
import numpy as np

from groundedvla.envs.libero_wrapper import LiberoWrapper, object_geoms
from groundedvla.eval.collisions import CollisionCounter, geoms_of_bodies
from groundedvla.eval.failure_labels import EpisodeTrace, label_failure
from groundedvla.eval.metrics import LIFT_HEIGHT_M, EpisodeResult, horizontal_error_cm

LEFT_FINGER_BODIES = ("gripper0_leftfinger", "gripper0_finger_joint1_tip")
RIGHT_FINGER_BODIES = ("gripper0_rightfinger", "gripper0_finger_joint2_tip")
COS_KNOCKED_OVER = np.cos(np.deg2rad(45))
JOINT_LIMIT_MARGIN_RAD = 0.01


class EpisodeMonitor:
    def __init__(self, env: LiberoWrapper):
        self.env, self.sim = env, env.sim
        self.target, self.dest_object = env.goal.target, env.goal.destination_object
        objects = env.scene_objects()
        self.movable = env.movable_objects()
        self.body = {n: self.sim.model.body_name2id(objects[n]) for n in self.movable}

        self.geom_owner: dict[int, str] = {}
        for name, root in objects.items():
            for g in object_geoms(self.sim, root):
                self.geom_owner[int(g)] = name
        self.left = geoms_of_bodies(self.sim, LEFT_FINGER_BODIES)
        self.right = geoms_of_bodies(self.sim, RIGHT_FINGER_BODIES)
        self.table = geoms_of_bodies(self.sim, ("table",))
        self.collisions = CollisionCounter.from_env(env)

        robot = env.robosuite_env.robots[0]
        self.joint_qpos = np.asarray(robot._ref_joint_pos_indexes)
        self.joint_range = self.sim.model._model.jnt_range[np.asarray(robot._ref_joint_indexes)]

        d = self.sim.data._data
        self.pos0 = {n: d.xpos[b].copy() for n, b in self.body.items()}
        self.up0 = d.xmat[self.body[self.target]].reshape(3, 3)[:, 2].copy()
        self.first_lifted: str | None = None
        self.target_lifted = False
        self.knocked_over = False
        self.max_displacement = 0.0
        self.hit_joint_limit = False
        self.eef: list[np.ndarray] = []

    def _gripped(self) -> set[str]:
        """Objects touching both fingers right now."""
        d = self.sim.data._data
        left, right = set(), set()
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if b in self.geom_owner:
                    if a in self.left:
                        left.add(self.geom_owner[b])
                    elif a in self.right:
                        right.add(self.geom_owner[b])
        return left & right

    def update(self, obs: dict) -> None:
        d = self.sim.data._data
        gripped = self._gripped()
        for n, b in self.body.items():
            if n in gripped and d.xpos[b][2] - self.pos0[n][2] >= LIFT_HEIGHT_M:
                self.first_lifted = self.first_lifted or n
                self.target_lifted |= n == self.target
            if n != self.target:
                self.max_displacement = max(self.max_displacement, float(np.linalg.norm(d.xpos[b] - self.pos0[n])))
        if not self.target_lifted:
            up = d.xmat[self.body[self.target]].reshape(3, 3)[:, 2]
            self.knocked_over |= float(up @ self.up0) < COS_KNOCKED_OVER
        q = d.qpos[self.joint_qpos]
        self.hit_joint_limit |= bool(np.any((q < self.joint_range[:, 0] + JOINT_LIMIT_MARGIN_RAD)
                                            | (q > self.joint_range[:, 1] - JOINT_LIMIT_MARGIN_RAD)))
        self.eef.append(np.asarray(obs["raw"]["robot0_eef_pos"]).copy())
        self.collisions.update()

    def _target_contacts(self) -> set:
        """What the target touches now: scene object names and/or the string "table"."""
        d = self.sim.data._data
        touching = set()
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if self.geom_owner.get(a) == self.target:
                    if b in self.table:
                        touching.add("table")
                    elif b in self.geom_owner and self.geom_owner[b] != self.target:
                        touching.add(self.geom_owner[b])
        return touching

    def _resting_on(self, touching: set) -> str | None:
        if self.target in self._gripped():
            return None
        z = self.env.object_center(self.target)[2]
        below = [n for n in touching - {"table"} if self.env.object_center(n)[2] < z]
        if self.dest_object in below:
            return self.dest_object
        return below[0] if below else None

    def _destination_center(self) -> np.ndarray:
        name = self.env.goal.destination
        if name in self.env.scene_objects():
            return self.env.object_center(name)
        return np.asarray(self.env.robosuite_env.object_states_dict[name].get_geom_state()["pos"])  # region site

    def trace(self) -> EpisodeTrace:
        touching = self._target_contacts()
        resting = self._resting_on(touching)
        return EpisodeTrace(
            first_lifted=self.first_lifted, target=self.target,
            max_nontarget_displacement_m=self.max_displacement,
            target_knocked_over_before_grasp=self.knocked_over, target_ever_lifted=self.target_lifted,
            target_on_table_outside_dest="table" in touching and resting is None and not self.env.eval_goal(),
            target_resting_on=resting, destination=self.dest_object,
            eef_positions=np.asarray(self.eef) if self.eef else np.zeros((0, 3)),
            hit_joint_limit=self.hit_joint_limit)

    def finish(self, res: EpisodeResult) -> None:
        """Fill the result's metric fields; `res.success` and `res.failure` ("aborted") are already set."""
        res.correct_object = self.first_lifted == self.target
        res.grasp_success = self.target_lifted
        res.correct_destination = self.env.eval_goal()
        res.final_pos_error_cm = horizontal_error_cm(self.env.object_center(self.target), self._destination_center())
        res.collisions = self.collisions.count
        res.first_lifted = self.first_lifted
        res.max_nontarget_displacement_cm = self.max_displacement * 100
        if not res.success and res.failure is None:
            res.failure = label_failure(self.trace())
