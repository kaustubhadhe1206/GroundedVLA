"""Instruction parsing and spatial target resolution in groundedvla/grounding/perception.py."""
import numpy as np
import pytest

from groundedvla.grounding.perception import ParsedInstruction, matches, nms, parse_instruction, resolve_target


@pytest.mark.parametrize("text, target, relation, anchors, dest", [
    ("pick up the black bowl between the plate and the ramekin and place it on the plate",
     "black bowl", "between", ["plate", "ramekin"], "plate"),
    ("pick up the black bowl next to the cookie box and place it on the plate",
     "black bowl", "next to", ["cookie box"], "plate"),
    ("pick up the black bowl on the cookie box and place it on the plate", "black bowl", "on", ["cookie box"], "plate"),
    ("pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
     "black bowl", "in", ["wooden cabinet"], "plate"),
    ("pick up the black bowl from table center and place it on the plate", "black bowl", "center", [], "plate"),
    ("put the bowl on the stove", "bowl", None, [], "stove"),
    ("put the bowl on the plate", "bowl", None, [], "plate"),
    ("put the bowl on top of the cabinet", "bowl", None, [], "cabinet"),
    ("Grab... no", None, None, None, None),
])
def test_parse_instruction(text, target, relation, anchors, dest):
    if target is None:
        with pytest.raises(ValueError):
            parse_instruction(text)
        return
    p = parse_instruction(text)
    assert (p.target, p.relation, p.anchors, p.destination) == (target, relation, anchors, dest)


def test_every_paraphrase_parses_like_its_original():
    import json

    from groundedvla.tasks import TASKS
    from groundedvla.utils.paths import REPO_ROOT

    originals = {"T1": "pick up the black bowl between the plate and the ramekin and place it on the plate",
                 "T2": "pick up the black bowl next to the cookie box and place it on the plate",
                 "T3": "put the bowl on the stove", "T4": "put the bowl on the plate"}
    para = json.loads((REPO_ROOT / "groundedvla/ood/paraphrases.json").read_text())
    for key in TASKS:
        want = parse_instruction(originals[key])
        for p in para[key]:
            got = parse_instruction(p)
            assert (got.target, got.relation, sorted(got.anchors), got.destination) == \
                   (want.target, want.relation, sorted(want.anchors), want.destination), p


def test_matches_uses_head_noun():
    assert matches("black bowl", "black bowl") and matches("bowl", "black bowl")
    assert not matches("plate", "black bowl") and matches("cookie box", "cookie box")


def test_nms_drops_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]))) == [0, 2]


def setup_scene():
    # detections: 0 bowl A, 1 bowl B, 2 plate, 3 ramekin, 4 cookie box
    centers = np.array([[0.50, 0.00, 0.95], [0.50, 0.20, 0.95], [0.40, -0.10, 0.92],
                        [0.60, 0.10, 0.93], [0.45, 0.25, 0.95]])
    scores = np.array([0.6, 0.9, 0.8, 0.7, 0.8])     # the wrong bowl has the higher detector score
    anchors = {"black bowl": 1, "plate": 2, "ramekin": 3, "cookie box": 4}
    return centers, scores, anchors


def test_between_picks_the_bowl_near_the_midpoint_not_the_top_score():
    centers, scores, anchors = setup_scene()
    p = ParsedInstruction("black bowl", "plate", "between", ["plate", "ramekin"])   # midpoint (0.5, 0.0)
    assert resolve_target(p, [0, 1], centers, scores, anchors) == 0


def test_next_to_picks_the_closest_bowl():
    centers, scores, anchors = setup_scene()
    p = ParsedInstruction("black bowl", "plate", "next to", ["cookie box"])
    assert resolve_target(p, [0, 1], centers, scores, anchors) == 1


def test_missing_anchor_falls_back_to_score():
    centers, scores, anchors = setup_scene()
    p = ParsedInstruction("black bowl", "plate", "between", ["plate", "teapot"])
    assert resolve_target(p, [0, 1], centers, scores, anchors | {"teapot": None}) == 1
