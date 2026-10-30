"""Unit tests for the Part A pipeline pieces that need no simulator or weights. Run: pytest tests -q"""
import numpy as np
import pytest

from groundedvla.policies.chunking import ChunkExecutor


def counting_policy():
    calls = {"n": 0}

    def predict(_obs):
        calls["n"] += 1
        base = calls["n"] * 100
        return np.arange(base, base + 50, dtype=np.float32)[:, None].repeat(7, 1)

    return predict, calls


def test_naive_executes_horizon_then_repredicts():
    predict, calls = counting_policy()
    ex = ChunkExecutor("naive", horizon=5)
    actions = [ex.act({}, predict)[0][0] for _ in range(12)]
    assert calls["n"] == 3
    assert actions[:5] == [100, 101, 102, 103, 104] and actions[5] == 200 and actions[10] == 300


def test_temporal_ensemble_weights_oldest_most():
    chunks = iter([np.zeros((50, 7)), np.ones((50, 7))])
    ex = ChunkExecutor("temporal_ensemble", te_m=0.01)
    a0, _ = ex.act({}, lambda o: next(chunks))
    a1, _ = ex.act({}, lambda o: next(chunks))
    assert np.allclose(a0, 0)
    w = np.exp(-0.01 * np.arange(2))
    assert np.allclose(a1, w[1] / w.sum())        # oldest chunk (zeros) has the larger weight
    assert a1[0] < 0.5


def test_temporal_ensemble_drops_expired_chunks():
    ex = ChunkExecutor("temporal_ensemble")
    for _ in range(5):
        ex.act({}, lambda o: np.ones((2, 7)))
    assert all(s + len(c) > ex.t - 1 for s, c in ex.chunks) and len(ex.chunks) == 2


def test_executor_rejects_bad_args():
    with pytest.raises(ValueError):
        ChunkExecutor("bogus")
    with pytest.raises(ValueError):
        ChunkExecutor("naive", horizon=0)


def test_quat2axisangle_matches_robosuite_near_pi():
    pytest.importorskip("robosuite")
    from robosuite.utils.transform_utils import quat2axisangle as rs_q2aa

    from groundedvla.envs.libero_wrapper import quat2axisangle

    rng = np.random.default_rng(0)
    quats = rng.normal(size=(200, 4))
    quats[:100, 3] = rng.uniform(-0.05, 0.05, 100)       # rotations near 180 deg, like the LIBERO gripper
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    for q in quats:
        assert np.allclose(quat2axisangle(q.copy()), rs_q2aa(q.copy()), atol=1e-6)


def test_openvla_image_preprocessing_shape():
    pytest.importorskip("PIL")
    from groundedvla.policies.openvla import openvla_image

    img = openvla_image(np.zeros((256, 256, 3), np.uint8))
    assert img.size == (224, 224) and img.mode == "RGB"
