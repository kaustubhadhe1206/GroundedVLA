"""Generate the position / scene OOD initial states once, with fixed seeds (plan Sections 6.1, 6.2).

All three variants are evaluated on exactly these states. They are saved in the repo
(groundedvla/data/ood_init_states.npz, small) and listed in split_manifest["ood"], so Colab needs
nothing from Drive. Viewpoint OOD is applied at reset (CameraPerturbation) and language OOD uses
paraphrases.json, so neither needs states.

Conditions (T1-T4; T5-T7 only get the viewpoint conditions, see configs/eval_ood.yaml):
  pos_small    every object on the table moved by 5 cm,      one random direction per episode
  pos_large    every object on the table moved by 10-15 cm,  one random direction per episode
  layout_swap  T1, T2: plate and ramekin trade places (the target bowl stays "between the plate and
               the ramekin" / "next to the cookie box", so the instruction is still true; the
               destination moves ~25 cm)
  distractor   one extra object (an alphabet-soup can, not confusable with a bowl or a plate) in a
               fixed empty spot away from the target and destination; the scene is a copy of the
               task's BDDL with that object added (groundedvla/ood/bddl/), the rest of the layout is
               sampled by LIBERO as for its own test states

pos_* and layout_swap start from LIBERO's 50 fixed test states (state i -> episode i) and move the
objects' free joints, so the state vector has the same layout and loads into the normal scene. All
objects move together: moving each one independently would break "between" / "next to" in T1/T2.
Objects resting on a fixture (T2's second bowl on the stove) stay with that fixture.

Validity (LIBERO starts objects ~7 cm above their resting height and lets them drop while settling):
after setting the state and the settle steps, every object must be within 1.5 cm (x, y) of where it
was put (beyond the slide the unmoved LIBERO state shows on its own), rest at its usual height (+-1 cm: not on top of another object) and not be tipped (>10 deg);
target and destination must be inside the reachable area and visible in the agentview image; the
goal must not already hold. Candidates are screened in a scene without depth/segmentation rendering
(fast), and every kept state is re-checked through the exact reset run_eval does.

Number of states per condition = its rollouts in configs/eval_ood.yaml (50 pos_large, 25 others).

    python -m groundedvla.ood.make_ood_states --tasks T1            # writes groundedvla/data/ood_parts/T1.npz
    python -m groundedvla.ood.make_ood_states --merge               # parts -> ood_init_states.npz + manifest
    python -m groundedvla.ood.make_ood_states --tasks T1 --conditions pos_large --n 3 --dry-run
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from groundedvla.envs.libero_wrapper import LiberoWrapper
from groundedvla.tasks import NUM_SETTLE_STEPS, TASKS
from groundedvla.utils.paths import REPO_ROOT

MANIFEST = REPO_ROOT / "groundedvla" / "data" / "split_manifest.json"
STATES_FILE = REPO_ROOT / "groundedvla" / "data" / "ood_init_states.npz"
PARTS_DIR = REPO_ROOT / "groundedvla" / "data" / "ood_parts"
BDDL_DIR = REPO_ROOT / "groundedvla" / "ood" / "bddl"
EVAL_OOD = REPO_ROOT / "configs" / "eval_ood.yaml"

CONDITIONS = ("pos_small", "pos_large", "layout_swap", "distractor")
SHIFT_M = {"pos_small": (0.05, 0.05), "pos_large": (0.10, 0.15)}
SWAP = {"T1": ("plate_1", "glazed_rim_porcelain_ramekin_1"), "T2": ("plate_1", "glazed_rim_porcelain_ramekin_1")}
DISTRACTOR = ("alphabet_soup_1", "alphabet_soup")
# Empty spot per scene, >= 15 cm from the target and destination (x y x y, table frame).
DISTRACTOR_REGION = {"libero_spatial": (-0.085, -0.01, -0.065, 0.01), "libero_goal": (0.09, 0.14, 0.11, 0.16)}

# Target / destination centers must stay where the arm can work and the camera looks (table frame, m).
# Training layouts span x -0.21..0.14, y -0.07..0.33; the robot base is at x = -0.66.
REACH_X, REACH_Y = (-0.32, 0.18), (-0.33, 0.35)
MAX_MOVE_M, MAX_DZ_M, MAX_TILT_DEG = 0.015, 0.01, 10.0
MIN_VISIBLE_PX = 40
MAX_TRIES = 40


# ---------------------------------------------------------------- state helpers (no simulator)
def qpos_index(env: LiberoWrapper) -> dict[str, int]:
    """{movable object: index of its free-joint x in the flattened sim state [time, qpos, qvel]}."""
    sim = env.sim
    return {n: 1 + int(sim.model.get_joint_qpos_addr(o.joints[-1])[0])
            for n, o in env.robosuite_env.objects_dict.items()}


def poses_in(state: np.ndarray, idx: dict[str, int]) -> dict[str, np.ndarray]:
    """{name: pose[7] (x y z qw qx qy qz)} read from a flattened state."""
    return {n: np.asarray(state[i:i + 7], dtype=np.float64) for n, i in idx.items()}


def with_xy(state: np.ndarray, idx: dict[str, int], xy: dict[str, np.ndarray]) -> np.ndarray:
    """Copy of the state with objects moved to new (x, y); everything else as in the base state."""
    s = np.array(state, dtype=np.float64)
    for n, p in xy.items():
        s[idx[n]:idx[n] + 2] = p
    return s


def tilt_deg(q_wxyz_a: np.ndarray, q_wxyz_b: np.ndarray) -> float:
    """Angle between the two orientations' z axes (yaw alone is not a tilt)."""
    ra, rb = (Rotation.from_quat(np.roll(q, -1)) for q in (q_wxyz_a, q_wxyz_b))
    za, zb = ra.apply([0, 0, 1]), rb.apply([0, 0, 1])
    return float(np.degrees(np.arccos(np.clip(za @ zb, -1, 1))))


def add_distractor(text: str, region_xyxy: tuple[float, float, float, float]) -> str:
    """BDDL text with DISTRACTOR declared and placed on the table in `region_xyxy`."""
    name, category = DISTRACTOR
    x0, y0, x1, y1 = region_xyxy
    region = (f"      (distractor_region\n          (:target main_table)\n          (:ranges (\n"
              f"              ({x0} {y0} {x1} {y1})\n            )\n          )\n      )\n")
    text = re.sub(r"(\(:regions\n)", lambda m: m.group(1) + region, text, count=1)
    text = re.sub(r"(\(:objects\n)", rf"\1    {name} - {category}\n", text, count=1)
    text = re.sub(r"(\(:init\n)", rf"\1    (On {name} main_table_distractor_region)\n", text, count=1)
    if text.count(name) != 2 or "distractor_region" not in text:
        raise ValueError("BDDL layout not as expected")
    return text


def rollouts_per_condition() -> dict[str, int]:
    conds = yaml.safe_load(EVAL_OOD.read_text())["conditions"]
    return {c: conds[c]["rollouts"] for c in CONDITIONS}


# ---------------------------------------------------------------- simulator checks
def table_objects(env: LiberoWrapper) -> list[str]:
    """Movable objects placed on the table itself (not in/on a fixture) by the BDDL init."""
    on = {s[1]: s[2] for s in env.robosuite_env.parsed_problem["initial_state"] if s[0] == "on"}
    return [n for n in env.movable_objects() if str(on.get(n, "")).startswith("main_table")]


def settle(env: LiberoWrapper, state: np.ndarray, exact: bool) -> dict:
    """exact: the reset run_eval does (scene rebuilt, state set, settle steps). Otherwise only set the
    state in the current scene and settle (same physics, no rebuild; used to screen candidates)."""
    if exact:
        return env.reset(state)
    env.env.set_init_state(state)
    e = env.robosuite_env
    e.timestep, e.done = 0, False            # many screenings in one episode would hit the horizon
    for _ in range(NUM_SETTLE_STEPS):
        raw, *_ = env.env.step(env.dummy_action())
    return {"raw": raw}


def resting(env: LiberoWrapper, state: np.ndarray, idx: dict[str, int]) -> dict[str, np.ndarray]:
    """Settled pose of every object from a known-good state (resting height and upright reference)."""
    settle(env, state, exact=False)
    return poses_in(np.asarray(env.sim.get_state().flatten()), idx)


def check(env: LiberoWrapper, state: np.ndarray, idx: dict[str, int], rest: dict[str, np.ndarray],
          exact: bool, drift: dict[str, np.ndarray] | None = None) -> str | None:
    """None if the state is valid, else a short reason. `exact` also checks visibility (needs seg).
    `drift`: how far each object slides while settling in the unmoved base state (some LIBERO test
    states slide a bowl by ~2 cm on their own); only movement beyond that counts as pushed."""
    placed = poses_in(state, idx)
    obs = settle(env, state, exact)
    now = poses_in(np.asarray(env.sim.get_state().flatten()), idx)
    for n, q in now.items():
        expected = placed[n][:2] + (drift[n] if drift and n in drift else 0)
        if np.linalg.norm(q[:2] - expected) > MAX_MOVE_M:
            return f"{n} pushed"
        if n in rest and abs(q[2] - rest[n][2]) > MAX_DZ_M:
            return f"{n} not resting on the table"
        if tilt_deg(q[3:], rest[n][3:] if n in rest else placed[n][3:]) > MAX_TILT_DEG:
            return f"{n} tipped"
    for n in (env.goal.target, env.goal.destination_object):
        c = env.object_center(n)
        if n in idx and not (REACH_X[0] <= c[0] <= REACH_X[1] and REACH_Y[0] <= c[1] <= REACH_Y[1]):
            return f"{n} out of reach"
    if env.eval_goal():
        return "goal already satisfied"
    if exact:
        from groundedvla.grounding.masks import instance_id_map

        ids = instance_id_map(env.robosuite_env)
        for n in (env.goal.target, env.goal.destination_object):
            if np.count_nonzero(obs["seg.agentview"] == ids[n]) < MIN_VISIBLE_PX:
                return f"{n} not visible"
    return None


class Checker:
    """A fast screening scene (no depth/seg) and an exact one (as in evaluation) for one task/scene."""

    def __init__(self, task_key: str, bddl_file: str | None = None, seed: int = 0):
        self.fast = LiberoWrapper(TASKS[task_key], with_depth_seg=False, bddl_file=bddl_file, seed=seed)
        self.exact = LiberoWrapper(TASKS[task_key], with_depth_seg=True, bddl_file=bddl_file, seed=seed)
        self.fast.reset()
        self.idx = qpos_index(self.fast)

    def __call__(self, state: np.ndarray, rest: dict[str, np.ndarray],
                 drift: dict[str, np.ndarray] | None = None) -> str | None:
        why = check(self.fast, state, self.idx, rest, exact=False, drift=drift)
        if why is None:
            why = check(self.exact, state, self.idx, rest, exact=True, drift=drift)
            if why is not None:
                why = f"exact reset: {why}"
        return why

    def close(self):
        self.fast.close()
        self.exact.close()


def count(rejected: dict, why: str) -> None:
    rejected[why] = rejected.get(why, 0) + 1


# ---------------------------------------------------------------- conditions
def shifted_states(task_key: str, cond: str, n: int, rng: np.random.Generator):
    """pos_small / pos_large / layout_swap from LIBERO's fixed test states."""
    chk = Checker(task_key)
    base_states = chk.fast.init_states()
    movers = table_objects(chk.fast)
    states, info, rejected = [], [], {}
    i = 0
    while len(states) < n and i < len(base_states):
        base = base_states[i]
        rest = resting(chk.fast, base, chk.idx)
        poses = poses_in(base, chk.idx)
        drift = {m: rest[m][:2] - poses[m][:2] for m in poses}
        why = check(chk.fast, base, chk.idx, rest, exact=False, drift=drift)
        if why is not None:                     # LIBERO's own state fails the checks: shifting won't help
            count(rejected, f"base state: {why}")
            i += 1
            continue
        for _ in range(MAX_TRIES if cond != "layout_swap" else 1):      # the swap is deterministic
            if cond == "layout_swap":
                a, b = SWAP[task_key]
                xy, offset = {a: poses[b][:2], b: poses[a][:2]}, np.zeros(2)   # each keeps its own height
            else:
                d, th = rng.uniform(*SHIFT_M[cond]), rng.uniform(0, 2 * np.pi)
                offset = d * np.array([np.cos(th), np.sin(th)])
                xy = {m: poses[m][:2] + offset for m in movers}
            state = with_xy(base, chk.idx, xy)
            why = chk(state, rest, drift)
            if why is None:
                states.append(state)
                info.append([i, *offset])
                break
            count(rejected, why)
        i += 1                                                           # a base state without a valid
    chk.close()                                                          # candidate is skipped
    return np.array(states), np.array(info, dtype=np.float64), rejected


def distractor_bddl(task_key: str) -> Path:
    """A copy of the task's BDDL with one extra object in a fixed empty region (written to the repo)."""
    from libero.libero import get_libero_path

    spec = TASKS[task_key]
    text = (Path(get_libero_path("bddl_files")) / spec.suite / f"{spec.bddl_name}.bddl").read_text()
    BDDL_DIR.mkdir(parents=True, exist_ok=True)
    out = BDDL_DIR / f"{task_key}_distractor.bddl"
    out.write_text(add_distractor(text, DISTRACTOR_REGION[spec.suite]))
    return out


def distractor_states(task_key: str, n: int, seed: int):
    bddl = distractor_bddl(task_key)
    chk = Checker(task_key, bddl_file=str(bddl), seed=seed)
    # Resting heights: the task's own objects from a LIBERO test state of the plain scene; the can's
    # from its first sample (it always stands alone in its region).
    plain = LiberoWrapper(TASKS[task_key], with_depth_seg=False)
    plain.reset()
    rest = resting(plain, plain.init_states()[0], qpos_index(plain))
    plain.close()
    states, info, rejected = [], [], {}
    sampler = chk.fast
    for i in range(n):
        for _ in range(MAX_TRIES):
            sampler.env.reset()                                          # LIBERO samples a layout
            state = np.asarray(sampler.sim.get_state().flatten())
            if DISTRACTOR[0] not in rest:
                rest = rest | {DISTRACTOR[0]: resting(sampler, state, chk.idx)[DISTRACTOR[0]]}
            why = chk(state, rest)
            if why is None:
                c = poses_in(state, chk.idx)[DISTRACTOR[0]]
                states.append(state)
                info.append([i, c[0], c[1]])
                break
            count(rejected, why)
        else:
            print(f"  {task_key} distractor ep{i}: no valid sample in {MAX_TRIES} tries, skipped")
    chk.close()
    return np.array(states), np.array(info, dtype=np.float64), rejected


# ---------------------------------------------------------------- main
def generate(tasks: list[str], conditions: list[str], n_override: int | None, seed: int) -> dict:
    per_cond = rollouts_per_condition()
    arrays = {}
    for task_key in tasks:
        ti = list(TASKS).index(task_key)
        for cond in conditions:
            if cond == "layout_swap" and task_key not in SWAP:
                continue
            n = n_override or per_cond[cond]
            t0 = time.time()
            if cond == "distractor":
                states, info, rejected = distractor_states(task_key, n, seed + 100 * ti)
            else:
                rng = np.random.default_rng([seed, CONDITIONS.index(cond), ti])
                states, info, rejected = shifted_states(task_key, cond, n, rng)
            print(f"{task_key} {cond}: {len(states)}/{n} states in {time.time() - t0:.0f}s, "
                  f"rejected {sum(rejected.values())}: {rejected}", flush=True)
            if len(states):
                arrays[f"{cond}/{task_key}"] = states
                arrays[f"{cond}/{task_key}/info"] = info
    return arrays


def merge(seed: int) -> None:
    arrays = dict(np.load(STATES_FILE)) if STATES_FILE.exists() else {}
    for part in sorted(PARTS_DIR.glob("*.npz")):
        arrays.update(dict(np.load(part)))
    manifest = json.loads(MANIFEST.read_text())
    ood = {}
    for key, states in sorted(arrays.items()):
        if key.endswith("/info"):
            continue
        cond, task_key = key.split("/")
        bddl = (BDDL_DIR / f"{task_key}_distractor.bddl").relative_to(REPO_ROOT).as_posix() \
            if cond == "distractor" else None
        entry = ood.setdefault(cond, {"file": STATES_FILE.relative_to(REPO_ROOT).as_posix(), "seed": seed,
                                      "generator": "groundedvla/ood/make_ood_states.py", "tasks": {}})
        entry["tasks"][task_key] = {"n": int(len(states)), "bddl": bddl}
    manifest["ood"] = ood
    np.savez_compressed(STATES_FILE, **arrays)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {STATES_FILE} ({STATES_FILE.stat().st_size / 1024:.0f} KB) and split_manifest['ood']:")
    for cond, e in ood.items():
        print(f"  {cond}: " + ", ".join(f"{t} {v['n']}" for t, v in e["tasks"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3", "T4"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--n", type=int, default=None, help="states per task and condition (default: eval_ood.yaml)")
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--merge", action="store_true", help="combine ood_parts/*.npz into the final file + manifest")
    ap.add_argument("--dry-run", action="store_true", help="generate and report, write nothing")
    args = ap.parse_args()
    if args.merge:
        merge(args.seed)
        return
    arrays = generate(args.tasks, args.conditions, args.n, args.seed)
    if args.dry_run:
        print("dry run: nothing written")
        return
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PARTS_DIR / f"{'_'.join(args.tasks)}.npz"
    np.savez_compressed(out, **arrays)
    print(f"wrote {out}; run --merge when all parts are done")


if __name__ == "__main__":
    main()
