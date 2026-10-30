"""Fast unit tests that need no simulator or model weights. Run: pytest tests -q"""
import numpy as np
import pytest
import torch

from groundedvla.data.bddl_parser import parse_bddl, region_to_object
from groundedvla.eval.failure_labels import EpisodeTrace, label_failure
from groundedvla.eval.metrics import wilson_ci
from groundedvla.grounding.grounding import GROUNDING_DIM, Normalizer
from groundedvla.grounding.masks import mask_image
from groundedvla.models.grounding_adapter import GroundingAdapter


def test_region_to_object():
    assert region_to_object("flat_stove_1_cook_region") == "flat_stove_1"
    assert region_to_object("plate_1") == "plate_1"


def test_parse_bddl(tmp_path):
    p = tmp_path / "t.bddl"
    p.write_text("(define (problem x)\n (:language Put the bowl on the stove)\n"
                 " (:goal\n    (And (On akita_black_bowl_1 flat_stove_1_cook_region))\n  )\n)")
    g = parse_bddl(p)
    assert (g.target, g.destination_object, g.instruction) == (
        "akita_black_bowl_1", "flat_stove_1", "Put the bowl on the stove")


def test_wilson_ci():
    lo, hi = wilson_ci(30, 50)
    assert 0.45 < lo < 0.6 < hi < 0.75


def test_mask_image():
    seg = np.zeros((4, 4), np.int32)
    seg[0, 0], seg[1, 1] = 3, 5
    img = mask_image(seg, 3, 5)
    assert img[0, 0].tolist() == [255, 0, 0] and img[1, 1].tolist() == [0, 255, 0]


def test_normalizer_range():
    x = np.random.uniform(-0.5, 0.8, (100, GROUNDING_DIM)).astype(np.float32)
    y = Normalizer.fit(x)(x)
    assert np.all(np.abs(y[:, :15]) <= 1 + 1e-5)


def test_adapter_shapes_and_zero_init():
    ad = GroundingAdapter(hidden_size=960, num_tokens=2)
    g = torch.randn(3, GROUNDING_DIM)
    out = ad(g)
    assert out.shape == (3, 2, 960)
    assert torch.all(out == 0)  # zero-init final layer
    out.sum().backward()
    assert ad.mlp[-1].weight.grad is not None


@pytest.mark.parametrize("first,expected", [("plate_1", "wrong_object"), (None, "failed_grasp")])
def test_failure_order(first, expected):
    tr = EpisodeTrace(first_lifted=first, target="bowl", max_nontarget_displacement_m=0.0,
                      target_knocked_over_before_grasp=False, target_ever_lifted=False,
                      target_on_table_outside_dest=False, target_resting_on=None, destination="plate_1",
                      eef_positions=np.random.rand(60, 3), hit_joint_limit=False)
    assert label_failure(tr) == expected
