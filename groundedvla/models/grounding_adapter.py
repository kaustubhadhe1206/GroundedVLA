"""Grounding adapter: 16-D grounding vector -> K tokens (Step B2).

15 coords x 8 Fourier freqs x (sin, cos) = 240, + 15 raw + 1 visibility = 256 inputs.
MLP 256 -> 512 -> 512 -> K*hidden with GELU + LayerNorm; final layer zero-initialised.
"""
import math

import torch
from torch import nn

from groundedvla.grounding.grounding import COORD_DIM, GROUNDING_DIM


class FourierFeatures(nn.Module):
    def __init__(self, num_freqs: int = 8):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(num_freqs)) * math.pi, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D) -> (B, D * 2F)
        xf = x[..., None] * self.freqs
        return torch.cat([xf.sin(), xf.cos()], dim=-1).flatten(-2)


class GroundingAdapter(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int = 2, num_freqs: int = 8, width: int = 512):
        super().__init__()
        self.num_tokens, self.hidden_size = num_tokens, hidden_size
        self.fourier = FourierFeatures(num_freqs)
        in_dim = COORD_DIM * 2 * num_freqs + COORD_DIM + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, num_tokens * hidden_size),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, g: torch.Tensor) -> torch.Tensor:  # (B, 16) -> (B, K, hidden)
        assert g.shape[-1] == GROUNDING_DIM
        coords, vis = g[..., :COORD_DIM], g[..., COORD_DIM:]
        x = torch.cat([self.fourier(coords), coords, vis], dim=-1)
        return self.mlp(x).view(*g.shape[:-1], self.num_tokens, self.hidden_size)
