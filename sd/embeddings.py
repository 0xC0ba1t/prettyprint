"""Diffusion timestep embeddings."""
import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal features followed by a two-layer MLP, producing the UNet's global conditioning vector."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(timesteps, self.in_dim))
