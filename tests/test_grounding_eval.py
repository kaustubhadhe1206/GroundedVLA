"""Unit tests for the A4-A7 pieces that need no simulator or weights. Run: pytest tests -q"""
from types import SimpleNamespace

import numpy as np

from groundedvla.data.replay_demos import is_noop
from groundedvla.envs import camera_utils as cu
from groundedvla.eval.collisions import CollisionCounter
from groundedvla.eval.failure_labels import EpisodeTrace, label_failure
from groundedvla.eval.report import add_degradation
from groundedvla.grounding.grounding import CameraView, GroundingTracker, compose, robust_center
from groundedvla.grounding.masks import instance_id_map


def test_instance_ids_are_one_based():
    env = SimpleNamespace(model=SimpleNamespace(instances_to_ids={"bowl": {}, "plate": {}}))
    assert instance_id_map(env) == {"bowl": 1, "plate": 2}


def test_backproject_then_project_roundtrip():
    K = np.array([[300.0, 0, 128], [0, 300.0, 128], [0, 0, 1]])
    pix = np.array([[10, 20], [128, 128], [200, 90]])
    depth = np.full((256, 256), 1.5)
    pts = cu.backproject(depth, pix, K)
    uv = pts @ K.T
    assert np.allclose(uv[:, :2] / uv[:, 2:3], pix[:, ::-1])


def test_opencv_to_policy_pixels_mirrors_columns():
    r, c = cu.opencv_to_policy_pixels(np.array([5]), np.array([10]), 256)
    assert r[0] == 5 and c[0] == 245


def test_robust_center_is_less_biased_than_median():
    # Most points on one side of a 10 cm object, as when a camera sees mostly one wall.
    x = np.concatenate([np.linspace(0, 0.02, 700), np.linspace(0.02, 0.10, 300)])
    pts = np.stack([x, x, x], axis=1)
    assert abs(robust_center(pts)[0] - 0.05) < 0.01
    assert abs(np.median(x) - 0.05) > 0.03


def test_camera_view_needs_enough_pixels():
    seg = np.zeros((32, 32), int)
    seg[:4, :4] = 1    # 16 pixels: below the limit
    seg[10:15, 10:15] = 2
    view = CameraView(seg, np.ones((32, 32)), np.array([[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]]), np.eye(4))
    assert view.centroid(1) is None
    assert view.centroid(2) is not None


def test_compose_layout():
    g = compose(np.array([1.0, 0, 0]), np.array([0, 2.0, 0]), np.array([0, 0, 3.0]), True)
    assert g.shape == (16,)
    assert g[9:12].tolist() == [1, 0, -3] and g[12:15].tolist() == [-1, 2, 0] and g[15] == 1


def test_tracker_starts_with_no_memory():
    t = GroundingTracker(1, 2)
    t.last_target = np.ones(3)
    t.reset()
    assert np.all(t.last_target == 0)


def test_is_noop():
    zero = np.array([0, 0, 0, 0, 0, 0, -1.0])
    assert not is_noop(zero, None)              # first action is always kept
    assert is_noop(zero, -1.0)
    assert not is_noop(zero, 1.0)               # gripper command changed
    assert not is_noop(np.array([0.1, 0, 0, 0, 0, 0, -1.0]), -1.0)


def test_collision_rules():
    c = CollisionCounter.__new__(CollisionCounter)
    c.robot, c.fingers, c.target, c.table, c.ignored = {1, 2}, {2}, {3}, {4}, {6}
    assert c._is_unintended(1, 5)               # robot hits another object
    assert c._is_unintended(5, 2)               # order does not matter, fingers count too
    assert not c._is_unintended(2, 3)           # touching the target is intended
    assert c._is_unintended(1, 4)               # arm link hits the table
    assert not c._is_unintended(2, 4)           # fingers may touch the table
    assert not c._is_unintended(1, 2)           # robot self contact
    assert not c._is_unintended(1, 6)           # mount is ignored


def _trace(**kw):
    base = dict(first_lifted=None, target="bowl", max_nontarget_displacement_m=0.0,
                target_knocked_over_before_grasp=False, target_ever_lifted=True,
                target_on_table_outside_dest=False, target_resting_on=None, destination="plate",
                eef_positions=np.random.default_rng(0).random((60, 3)), hit_joint_limit=False)
    return EpisodeTrace(**(base | kw))


def test_failure_labels_follow_plan_order():
    assert label_failure(_trace(first_lifted="other", max_nontarget_displacement_m=0.2)) == "wrong_object"
    assert label_failure(_trace(max_nontarget_displacement_m=0.06, target_ever_lifted=False)) == "collision"
    assert label_failure(_trace(target_knocked_over_before_grasp=True)) == "collision"
    assert label_failure(_trace(target_ever_lifted=False)) == "failed_grasp"
    assert label_failure(_trace(target_on_table_outside_dest=True)) == "dropped_object"
    assert label_failure(_trace(target_resting_on="stove")) == "wrong_destination"
    assert label_failure(_trace(eef_positions=np.zeros((60, 3)))) == "motion_failure"
    assert label_failure(_trace(hit_joint_limit=True)) == "motion_failure"
    assert label_failure(_trace()) == "timeout"


def test_degradation():
    rows = [{"label": "v1", "task": "T1", "condition": "in_distribution", "success": 0.9},
            {"label": "v1", "task": "T1", "condition": "yaw_p20", "success": 0.6}]
    add_degradation(rows)
    assert rows[0]["degradation"] == 0.0 and abs(rows[1]["degradation"] - 0.3) < 1e-9
