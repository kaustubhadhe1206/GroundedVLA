"""Contact-based unintended collision counting (Step A4).

Counts *new* contact pairs (absent in the previous control step, normal force > 1 N) between
  - any robot geom and any object other than the target, and
  - arm links (not fingers) and the table.
"""
import mujoco
import numpy as np

FORCE_THRESHOLD_N = 1.0


class CollisionCounter:
    def __init__(self, sim, robot_geoms: set[int], finger_geoms: set[int],
                 target_geoms: set[int], table_geoms: set[int]):
        """TODO(A4): build geom-id sets from sim.model names (robot0_*, gripper0_*, table)."""
        self.sim = sim
        self.robot, self.fingers = robot_geoms, finger_geoms
        self.target, self.table = target_geoms, table_geoms
        self.prev_pairs: set[tuple[int, int]] = set()
        self.count = 0

    def _is_unintended(self, g1: int, g2: int) -> bool:
        for a, b in ((g1, g2), (g2, g1)):
            if a in self.robot:
                if b in self.table:
                    return a not in self.fingers
                if b not in self.robot and b not in self.target:
                    return True
        return False

    def update(self) -> int:
        """Call once per control step. Returns number of new unintended contacts this step."""
        m, d = self.sim.model._model, self.sim.data._data
        pairs, force = set(), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, force)
            if abs(force[0]) > FORCE_THRESHOLD_N and self._is_unintended(c.geom1, c.geom2):
                pairs.add(tuple(sorted((c.geom1, c.geom2))))
        new = len(pairs - self.prev_pairs)
        self.prev_pairs = pairs
        self.count += new
        return new
