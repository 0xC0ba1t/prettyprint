"""Attention layers: single-head VAE attention and the UNet's text-conditioned transformer."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import group_norm
from typing import Optional


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


class BasicTransformerBlock(nn.Module):
    """self-attn -> cross-attn (text) -> GEGLU feed-forward, each with pre-LayerNorm + residual."""

    def __init__(self, dim: int, heads: int, head_dim: int, context_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, None, heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads, head_dim, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.ff(self.norm3(x))


class SpatialTransformer(nn.Module):
    """Applies transformer blocks over the flattened feature map of a (B, C, H, W) tensor."""

    def __init__(self, channels: int, head_dim: int, context_dim: int, depth: int = 1, groups: int = 32, dropout: float = 0.0):
        super().__init__()
        heads = max(1, channels // head_dim)
        inner = heads * head_dim
        self.norm = group_norm(channels, groups)
        self.proj_in = nn.Conv2d(channels, inner, 1)
        self.blocks = nn.ModuleList(
            [BasicTransformerBlock(inner, heads, head_dim, context_dim, dropout) for _ in range(depth)]
        )
        self.proj_out = nn.Conv2d(inner, channels, 1)
        nn.init.zeros_(self.proj_out.weight)  # block starts as identity, stabilising early training
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.proj_in(self.norm(x)).flatten(2).transpose(1, 2)  # (b, hw, inner)
        for block in self.blocks:
            y = block(y, context)
        y = y.transpose(1, 2).reshape(b, -1, h, w)
        return x + self.proj_out(y)
