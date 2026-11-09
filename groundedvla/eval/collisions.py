"""Contact-based unintended collision counting (Step A4).

Counts *new* contact pairs (absent in the previous control step, normal force > 1 N) between
  - any robot geom and any object other than the target, and
  - arm links (not fingers) and the table.
Contacts already present when the counter is created (the settled scene) are not counted, and the
robot's mount is ignored: it permanently touches the floor and the arm can only graze it.
"""
import mujoco
import numpy as np

from groundedvla.envs.libero_wrapper import body_descendants

FORCE_THRESHOLD_N = 1.0
FINGER_BODIES = ("gripper0_leftfinger", "gripper0_finger_joint1_tip",
                 "gripper0_rightfinger", "gripper0_finger_joint2_tip")


def geoms_of_bodies(sim, names) -> set[int]:
    m = sim.model._model
    return {int(g) for g in np.flatnonzero(np.isin(m.geom_bodyid, [sim.model.body_name2id(n) for n in names]))}


def geoms_with_body_prefix(sim, prefixes: tuple[str, ...]) -> set[int]:
    names = [sim.model.body_id2name(b) for b in range(sim.model.nbody)]
    return geoms_of_bodies(sim, [n for n in names if n and n.startswith(prefixes)])


class CollisionCounter:
    def __init__(self, sim, robot_geoms: set[int], finger_geoms: set[int], target_geoms: set[int],
                 table_geoms: set[int], ignored_geoms: set[int] = frozenset()):
        self.sim = sim
        self.robot, self.fingers = robot_geoms, finger_geoms
        self.target, self.table, self.ignored = target_geoms, table_geoms, ignored_geoms
        self.prev_pairs: set[tuple[int, int]] = self._current_pairs()
        self.count = 0

    @classmethod
    def from_env(cls, env) -> "CollisionCounter":
        """Geom sets for the current model. LIBERO rebuilds it on reset, so call this after reset."""
        sim = env.sim
        target_root = sim.model.body_name2id(env.scene_objects()[env.goal.target])
        m = sim.model._model
        target = {int(g) for g in np.flatnonzero(np.isin(m.geom_bodyid, body_descendants(m, target_root)))}
        return cls(sim, geoms_with_body_prefix(sim, ("robot0_", "gripper0_")),
                   geoms_of_bodies(sim, FINGER_BODIES), target, geoms_of_bodies(sim, ("table",)),
                   geoms_with_body_prefix(sim, ("mount0_",)))

    def _is_unintended(self, g1: int, g2: int) -> bool:
        if g1 in self.ignored or g2 in self.ignored:
            return False
        for a, b in ((g1, g2), (g2, g1)):
            if a in self.robot:
                if b in self.table:
                    return a not in self.fingers
                if b not in self.robot and b not in self.target:
                    return True
        return False

    def _current_pairs(self) -> set[tuple[int, int]]:
        m, d = self.sim.model._model, self.sim.data._data
        pairs, force = set(), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, force)
            if abs(force[0]) > FORCE_THRESHOLD_N and self._is_unintended(int(c.geom1), int(c.geom2)):
                pairs.add(tuple(sorted((int(c.geom1), int(c.geom2)))))
        return pairs

    def update(self) -> int:
        """Call once per control step. Returns number of new unintended contacts this step."""
        pairs = self._current_pairs()
        new = len(pairs - self.prev_pairs)
        self.prev_pairs = pairs
        self.count += new
        return new
