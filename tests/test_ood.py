"""OOD state helpers (ood/make_ood_states.py) and their use in run_eval, without a simulator."""
import json

import numpy as np
import pytest

from groundedvla.data.bddl_parser import parse_bddl
from groundedvla.eval import run_eval
from groundedvla.ood import make_ood_states as M

BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Put the bowl on the plate)
    (:regions
      (plate_region
          (:target main_table)
          (:ranges (
              (0.05 0.19 0.07 0.21)
            )
          )
      )
    )
  (:fixtures
    main_table - table
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
    plate_1 - plate
  )
  (:obj_of_interest
    akita_black_bowl_1
    plate_1
  )
  (:init
    (On akita_black_bowl_1 main_table_plate_region)
    (On plate_1 main_table_plate_region)
  )
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
)
"""


def test_with_xy_moves_only_xy_of_named_objects():
    state = np.arange(30, dtype=np.float64)
    idx = {"a": 1, "b": 8}
    out = M.with_xy(state, idx, {"a": np.array([100.0, 200.0])})
    assert out[1] == 100 and out[2] == 200
    assert np.array_equal(np.delete(out, [1, 2]), np.delete(state, [1, 2]))
    assert np.array_equal(M.poses_in(out, idx)["b"], state[8:15])
    assert state[1] == 1                                     # input untouched


def test_tilt():
    q = np.array([1.0, 0, 0, 0])
    assert M.tilt_deg(q, q) == pytest.approx(0, abs=1e-6)
    q90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])   # 90 deg about x
    assert M.tilt_deg(q, q90) == pytest.approx(90, abs=1e-4)
    qyaw = np.array([np.cos(0.5), 0, 0, np.sin(0.5)])               # yaw only: not a tilt
    assert M.tilt_deg(q, qyaw) == pytest.approx(0, abs=1e-4)


def test_add_distractor_keeps_goal_and_adds_object(tmp_path):
    text = M.add_distractor(BDDL, (0.1, 0.1, 0.12, 0.12))
    assert "alphabet_soup_1 - alphabet_soup" in text
    assert "(On alphabet_soup_1 main_table_distractor_region)" in text
    f = tmp_path / "x.bddl"
    f.write_text(text)
    g0, g1 = tmp_path / "orig.bddl", f
    g0.write_text(BDDL)
    a, b = parse_bddl(g0), parse_bddl(g1)
    assert (a.target, a.destination_object) == (b.target, b.destination_object)
    with pytest.raises(ValueError):
        M.add_distractor("no sections here", (0, 0, 1, 1))


def test_distractor_regions_far_from_targets():
    """The fixed distractor spot is >= 12 cm from every target / destination object in the training layouts."""
    centers = {"libero_spatial": [(0.053, 0.206), (-0.063, 0.202), (0.138, -0.071), (0.069, 0.195)],
               "libero_goal": [(-0.094, -0.002), (0.049, -0.026), (-0.308, 0.202)]}
    for suite, pts in centers.items():
        x0, y0, x1, y1 = M.DISTRACTOR_REGION[suite]
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        assert min(np.linalg.norm(c - np.array(p)) for p in pts) >= 0.12, suite


def test_run_eval_reads_ood_states(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "REPO_ROOT", tmp_path)
    arr = np.ones((3, 5))
    np.savez(tmp_path / "s.npz", **{"pos_large/T1": arr})
    manifest = json.loads(json.dumps({"ood": {"pos_large": {"file": "s.npz", "tasks": {"T1": {"n": 3}}}}}))
    assert np.array_equal(run_eval.ood_init_states(manifest, "pos_large", "T1"), arr)
