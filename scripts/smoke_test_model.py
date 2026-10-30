"""Load SmolVLA from the local D-drive copy and run one forward pass on random inputs.

Usage:
    python scripts/smoke_test_model.py [--grounded]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from groundedvla.utils.paths import load_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounded", action="store_true")
    a = ap.parse_args()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from groundedvla.models.grounded_smolvla import GroundedSmolVLAPolicy

    cls = GroundedSmolVLAPolicy if a.grounded else SmolVLAPolicy
    path = load_paths()["smolvla_base"]
    policy = cls.from_pretrained(str(path))
    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"{cls.__name__}: {n_total / 1e6:.1f}M params, {n_train / 1e6:.1f}M trainable")
    print("input features:", list(policy.config.input_features))
    print("chunk_size:", policy.config.chunk_size, "| cuda:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
