"""Step B2 "done when" checks for the V3 grounding adapter (runs on the laptop GPU, ~3 min).

1. Identity: V3 with grounding disabled gives the actions of plain SmolVLA (V2 inputs, same weights,
   same noise). Checked in fp32 (tolerance 1e-4): in bf16 the two extra (masked) tokens change the
   matmul shapes, so rounding differs by ~1 bf16 ulp per layer and the 10 denoising steps amplify it
   (measured Nov 15: 0.35 max |diff| in bf16 vs 3.6e-7 in fp32). Masking is exact; rounding is not.
2. Gradients: one training forward/backward on a real V3 batch reaches the adapter (at step 0 only
   its last layer gets gradient, because that layer starts at zero).
3. With grounding enabled the actions change once the adapter is non-zero.

    python scripts/smoke_test_b2.py
(Rollouts of V2/V3 checkpoints through run_sim are checked separately; see the plan, Step B2.)
"""
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402

from groundedvla.train import train as T  # noqa: E402
from groundedvla.utils.paths import REPO_ROOT, load_paths  # noqa: E402


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = load_paths()
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "v3_grounded.yaml").read_text())
    train_ds, _ = T.make_datasets(cfg, T.dataset_root(cfg["data"]["repo_id"], paths), None)
    init_name, base = T.init_weights(cfg, paths)
    pc = T.make_policy_config(cfg, base, train_ds.meta, "cuda")
    pre, _ = make_pre_post_processors(pc, dataset_stats=T.normalization_stats(train_ds.meta.stats, init_name, base))
    loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    raw = next(iter(loader))
    assert "observation.grounding" in raw and raw["observation.grounding"].shape == (2, 16)

    def batch():
        b = pre({k: (v.clone() if torch.is_tensor(v) else v) for k, v in raw.items()})
        assert "observation.grounding" in b, "the preprocessor dropped observation.grounding"
        return b

    noise = torch.randn(2, pc.chunk_size, pc.max_action_dim, device="cuda")

    # 1. plain SmolVLA with the same weights and inputs (V2 = 3 cameras, no grounding)
    plain = T.policy_class("v2").from_pretrained(str(base), config=pc).to("cuda").float().eval()
    with torch.no_grad():
        ref = plain.predict_action_chunk(batch(), noise=noise.clone()).float()
    del plain
    torch.cuda.empty_cache()

    v3 = T.policy_class("v3").from_pretrained(str(base), config=pc, **T.policy_kwargs(cfg)).to("cuda").float().eval()
    v3.grounding_enabled = False
    with torch.no_grad():
        off = v3.predict_action_chunk(batch(), noise=noise.clone()).float()
    diff = (off - ref).abs().max().item()
    print(f"[1] V3 grounding disabled vs plain SmolVLA (fp32): max |diff| = {diff:.2e}")
    assert diff < 1e-4, "V3 with grounding disabled must reproduce V2"

    # 2. gradients reach the adapter
    v3.grounding_enabled = True
    v3.train()
    v3.zero_grad(set_to_none=True)
    loss, _ = v3.forward(batch(), noise=noise.clone())
    loss.backward()
    last = v3.model.grounding_adapter.mlp[-1]
    g_last = last.weight.grad.norm().item() if last.weight.grad is not None else 0.0
    groups = T.param_groups(v3, cfg, 1e-4)
    print(f"[2] loss {loss.item():.4f}; adapter last-layer grad norm {g_last:.3e}; "
          f"param groups: {[len(g['params']) for g in groups]} (adapter lr {groups[-1]['lr']:.0e})")
    assert g_last > 0, "no gradient reached the grounding adapter"

    # 3. a non-zero adapter changes the actions (the tokens are really attended to)
    with torch.no_grad():
        last.weight.normal_(0, 0.02)
    v3.eval()
    with torch.no_grad():
        on = v3.predict_action_chunk(batch(), noise=noise.clone()).float()
    print(f"[3] adapter non-zero, grounding on vs off: max |diff| = {(on - off).abs().max().item():.3e}")
    assert (on - off).abs().max().item() > 0
    print("B2 checks passed")


if __name__ == "__main__":
    main()
