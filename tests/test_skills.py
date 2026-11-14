"""Free-form prompt -> trained skill matching used by the demo (groundedvla/demo/skills.py)."""
import pytest

from groundedvla.demo.skills import match_prompt, skills_in_scene

KITCHEN = {"T3": "put the bowl on the stove", "T4": "put the bowl on the plate",
           "T6": "open the top drawer and put the bowl inside", "T7": "put the wine bottle on top of the cabinet"}
T1 = {"T1": "pick up the black bowl between the plate and the ramekin and place it on the plate"}


def test_kitchen_tasks_share_one_scene():
    assert skills_in_scene("T3") == ["T3", "T4", "T6", "T7"]
    assert skills_in_scene("T1") == ["T1"] and skills_in_scene("T5") == ["T5"]


@pytest.mark.parametrize("prompt, task", [
    ("put bowl on stove", "T3"),
    ("Place the bowl onto the plate.", "T4"),
    ("stick the bowl into the top drawer", "T6"),
    ("open the drawer and put the bowl in it", "T6"),
    ("place the bottle on the cabinet", "T7"),
    ("set the wine bottle down on top of the cabinet", "T7"),
    ("put the bowl on top of the cabinet", None),     # bowl -> cabinet: not a trained skill
    ("put the wine bottle on the stove", None),
    ("hello robot", None),
])
def test_kitchen_prompts(prompt, task):
    m = match_prompt(prompt, KITCHEN)
    assert m.task == task, m.reason
    if task:
        assert m.instruction == KITCHEN[task]


def test_spatial_relation_must_name_the_same_objects():
    assert match_prompt("grab the bowl between the plate and the ramekin and put it on the plate", T1).task == "T1"
    assert match_prompt("put the black bowl between plate and ramekin on the plate", T1).task == "T1"
    assert match_prompt("pick up the black bowl next to the cookie box and place it on the plate", T1).task is None
