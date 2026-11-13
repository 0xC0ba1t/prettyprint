"""Attention layers: single-head VAE attention and the UNet's text-conditioned transformer."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import group_norm


class VAEAttnBlock(nn.Module):
    """Single-head self-attention over all spatial positions (used in the VAE mid block)."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.norm = group_norm(channels, groups)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = (t.transpose(1, 2).unsqueeze(1) for t in qkv.unbind(1))  # (b, 1, hw, c)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.squeeze(1).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)
