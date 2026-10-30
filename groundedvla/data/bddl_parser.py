"""Extract instruction, target, and destination from a LIBERO BDDL file (plan Section 3.2).

The goal predicate `(On <target> <destination>)` gives the language-to-object mapping used
by the mask (V2) and grounding (V3) inputs. The policy itself still only sees the instruction.
"""
import re
from dataclasses import dataclass
from pathlib import Path

_GOAL_RE = re.compile(r"\(:goal\s*\(And\s*\((On|In)\s+([^\s()]+)\s+([^\s()]+)\)\s*\)\s*\)", re.S)
_LANG_RE = re.compile(r"\(:language\s+([^)]+)\)")
_REGION_SUFFIXES = ("_cook_region", "_contain_region", "_top_region", "_region")


@dataclass(frozen=True)
class TaskGoal:
    instruction: str
    predicate: str          # "On" or "In"
    target: str             # e.g. akita_black_bowl_1
    destination: str        # raw goal argument, e.g. flat_stove_1_cook_region
    destination_object: str  # object that owns the region, e.g. flat_stove_1


def region_to_object(name: str) -> str:
    """Map a BDDL region/site name to the object instance that owns it.

    Example: flat_stove_1_cook_region -> flat_stove_1. Plain object names pass through.
    TODO(A6): verify against env.instance_to_id for every selected task.
    """
    for suffix in _REGION_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_bddl(path: str | Path) -> TaskGoal:
    text = Path(path).read_text()
    lang = _LANG_RE.search(text)
    goal = _GOAL_RE.search(text)
    if lang is None or goal is None:
        raise ValueError(f"{path}: goal is not a single On/In predicate")
    pred, target, dest = goal.groups()
    return TaskGoal(
        instruction=lang.group(1).strip(),
        predicate=pred,
        target=target,
        destination=dest,
        destination_object=region_to_object(dest),
    )


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        print(parse_bddl(p))
