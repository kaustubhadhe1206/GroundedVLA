"""Generate OOD initial states once, with fixed seeds, into the split manifest (Section 6).

Conditions: position small/large shift (edit BDDL placement regions), layout swap (T1, T2),
extra distractor. Viewpoint OOD is handled at reset by CameraPerturbation; language OOD by
paraphrases.json. Overlapping or fallen objects are resampled.

Usage:
    python -m groundedvla.ood.make_ood_states --tasks T1 T2 T3 --seed 3000
"""
import argparse
import re

import numpy as np

SHIFTS_M = {"pos_small": (0.05, 0.05), "pos_large": (0.10, 0.15)}
_RANGE_RE = re.compile(r"\(:ranges \(\s*\(([-\d.e ]+)\)\s*\)\s*\)")


def shift_regions(bddl_text: str, dist_range: tuple[float, float], rng: np.random.Generator) -> str:
    """Shift every placement region (x_min y_min x_max y_max) by dist in a random direction."""
    def _shift(m):
        x0, y0, x1, y1 = map(float, m.group(1).split())
        d, th = rng.uniform(*dist_range), rng.uniform(0, 2 * np.pi)
        dx, dy = d * np.cos(th), d * np.sin(th)
        return f"(:ranges (\n              ({x0 + dx} {y0 + dy} {x1 + dx} {y1 + dy})\n            )\n          )"
    return _RANGE_RE.sub(_shift, bddl_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--seed", type=int, default=3000)
    args = ap.parse_args()
    # TODO(Week 3):
    #  1. For each task/condition, write a shifted BDDL to outputs/ood_bddl/.
    #  2. Build an env from it, sample N init states (env.reset -> sim.get_state().flatten()).
    #  3. Reject states with overlapping objects or objects below table height.
    #  4. Save states (npz under datasets/ood_states) and settings into split_manifest["ood"].
    raise NotImplementedError


if __name__ == "__main__":
    main()
