"""Step B2 pieces that run without a GPU or the VLM (the full checks: scripts/smoke_test_b2.py)."""
import numpy as np
import pytest
import torch

from groundedvla.grounding.grounding import GROUNDING_DIM, compose
from groundedvla.models.grounding_adapter import GroundingAdapter
from groundedvla.policies.smolvla import noisy_grounding
from groundedvla.train.train import param_groups, policy_kwargs

V3_CFG = {"model": {"use_grounding": True, "grounding": {"num_tokens": 3, "fourier_freqs": 6, "mlp_width": 64,
                                                       "adapter_lr": 5e-5, "injection": "tokens"}}}


def test_adapter_shape_and_zero_init():
    a = GroundingAdapter(hidden_size=32, num_tokens=2, num_freqs=8, width=64)
    out = a(torch.randn(4, GROUNDING_DIM))
    assert out.shape == (4, 2, 32)
    assert torch.count_nonzero(out) == 0          # starts as a no-op (behaves like V2)


def test_noisy_grounding_keeps_vector_consistent():
    g = compose(np.array([0.5, 0.1, 0.0]), np.array([0.6, -0.1, 0.02]), np.array([0.3, 0.0, 0.3]), True)
    assert np.array_equal(noisy_grounding(g, 0.0, np.random.default_rng(0)), g)
    n = noisy_grounding(g, 0.03, np.random.default_rng(0))
    assert not np.allclose(n[0:6], g[0:6])
    assert np.allclose(n[6:9], g[6:9]) and n[15] == g[15]                 # eef and visibility untouched
    assert np.allclose(n[9:12], n[0:3] - n[6:9]) and np.allclose(n[12:15], n[3:6] - n[0:3])
    big = np.stack([noisy_grounding(g, 0.03, np.random.default_rng(i))[0] - g[0] for i in range(2000)])
    assert abs(big.std() - 0.03) < 0.003


def test_v3_policy_kwargs_and_adapter_param_group():
    assert policy_kwargs({"model": {"use_grounding": False}}) == {}
    assert policy_kwargs(V3_CFG) == {"num_grounding_tokens": 3, "fourier_freqs": 6, "mlp_width": 64}
    with pytest.raises(NotImplementedError):
        policy_kwargs({"model": {"use_grounding": True, "grounding": {"injection": "state_concat"}}})

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.frozen = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
            self.adapter = GroundingAdapter(8, num_tokens=1, num_freqs=2, width=8)
            self.frozen.requires_grad_(False)

        def adapter_parameters(self):
            return list(self.adapter.parameters())

    groups = param_groups(Fake(), V3_CFG, 1e-4)
    assert [g["lr"] for g in groups] == [1e-4, 5e-5]
    assert len(groups[0]["params"]) == 2 and len(groups[1]["params"]) == len(Fake().adapter_parameters())
    assert len(param_groups(torch.nn.Linear(2, 2), {"model": {}}, 1e-4)) == 1


def test_goal_objects_of_kitchen_skills():
    from groundedvla.demo.skills import goal_objects

    assert goal_objects("T3") == ("akita_black_bowl_1", "flat_stove_1")
    assert goal_objects("T4")[1] == "plate_1"
    assert goal_objects("T6")[0] == "akita_black_bowl_1"
