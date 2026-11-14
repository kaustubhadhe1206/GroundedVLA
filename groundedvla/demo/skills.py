"""Map a free-form prompt to a trained skill in the current scene (demo only).

The policies only learn the tasks they were trained on, so a free-form prompt works only if it asks
for one of those skills. This module makes that explicit instead of hoping the policy copes:

  1. skills_in_scene: the trained tasks that share the current scene. All LIBERO-Goal tasks use one
     kitchen with the same objects (T3, T4, T6, T7); each LIBERO-Spatial task has its own layout.
  2. match_prompt: parse the prompt into (target, destination) with the perception parser and compare
     the head nouns with every phrasing of every skill (instruction + paraphrases). Both must agree:
     "put the bowl on the cabinet" names a bowl and a cabinet, which no trained skill does, so it is
     reported as unmatched rather than mapped to the nearest bowl task.
  3. The demo then feeds the skill's trained instruction to the policy and checks that skill's own
     BDDL goal for success, and says so on the page.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from groundedvla.grounding.perception import parse_instruction
from groundedvla.tasks import TASKS
from groundedvla.utils.paths import REPO_ROOT

PARAPHRASES = REPO_ROOT / "groundedvla" / "ood" / "paraphrases.json"
SHARED_SCENE_SUITES = ("libero_goal",)        # one scene, many goals


@dataclass
class SkillMatch:
    task: str | None           # matched task key, or None
    instruction: str           # trained instruction of the match (or the prompt as typed)
    score: float               # 1.0 = target and destination agree
    reason: str


def skills_in_scene(task_key: str) -> list[str]:
    suite = TASKS[task_key].suite if task_key in TASKS else None
    if suite in SHARED_SCENE_SUITES:
        return [k for k, t in TASKS.items() if t.suite == suite]
    return [task_key] if task_key in TASKS else []


@lru_cache(maxsize=None)
def phrasings(task_key: str, instruction: str) -> tuple[str, ...]:
    try:
        para = json.loads(PARAPHRASES.read_text()).get(task_key, [])
    except (OSError, ValueError):
        para = []
    return (instruction, *[p for p in para if p and p != "TODO"])


def _head(noun: str) -> str:
    return noun.split()[-1]


def match_prompt(prompt: str, skills: dict[str, str]) -> SkillMatch:
    """skills: {task key: trained instruction} available in the current scene."""
    try:
        want = parse_instruction(prompt)
    except ValueError:
        return SkillMatch(None, prompt, 0.0, "couldn't find an object and a destination in the prompt")
    best: SkillMatch | None = None
    for key, instruction in skills.items():
        for phr in phrasings(key, instruction):
            try:
                have = parse_instruction(phr)
            except ValueError:
                continue
            score = 0.5 * (_head(want.target) == _head(have.target)) + \
                0.5 * (_head(want.destination) == _head(have.destination))
            # "the bowl next to the cookie box" is not the trained "bowl between the plate and the
            # ramekin": a spatial relation that names different objects is a different skill
            if want.relation and have.relation and \
                    {_head(a) for a in want.anchors} != {_head(a) for a in have.anchors}:
                score -= 0.25
            if best is None or score > best.score:
                best = SkillMatch(key, instruction, score, f"{want.target} -> {want.destination}")
    if best is None or best.score < 1.0:
        return SkillMatch(None, prompt, best.score if best else 0.0,
                          f"no trained skill moves the {want.target} to the {want.destination} in this scene")
    return best


@lru_cache(maxsize=None)
def task_instruction(task_key: str) -> str:
    """The benchmark instruction of a task (what the policies were trained on), without an env."""
    from libero.libero import benchmark

    t = TASKS[task_key]
    suite = benchmark.get_benchmark_dict()[t.suite]()
    return next(x.language for x in suite.tasks if x.name == t.bddl_name)


@lru_cache(maxsize=None)
def goal_state(task_key: str) -> list:
    """The task's BDDL goal as LIBERO's predicate lists, evaluable in any env of the same scene."""
    from libero.libero import get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    t = TASKS[task_key]
    path = Path(get_libero_path("bddl_files")) / t.suite / f"{t.bddl_name}.bddl"
    return robosuite_parse_problem(str(path))["goal_state"]
