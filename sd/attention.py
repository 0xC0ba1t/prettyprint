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


class CrossAttention(nn.Module):
    """Multi-head attention; self-attention when `context` is None, cross-attention otherwise."""

    def __init__(self, query_dim: int, context_dim: int = None, heads: int = 8, head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        inner = heads * head_dim
        context_dim = context_dim or query_dim
        self.heads = heads
        self.dropout = dropout
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        context = x if context is None else context
        b, n, _ = x.shape
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = (t.reshape(b, -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class GEGLU(nn.Module):
    """Gated GELU projection: (x W) * gelu(x V)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(GEGLU(dim, dim * mult), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
