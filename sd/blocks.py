"""Convolutional building blocks shared by the VAE and the UNet."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    """GroupNorm that falls back to the largest valid group count for small widths."""
    groups = min(groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """norm-silu-conv x2 with an optional additive time embedding and a 1x1 skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 0,
        dropout: float = 0.0,
        groups: int = 32,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = group_norm(in_channels, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None
        self.norm2 = group_norm(out_channels, groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class Downsample(nn.Module):
    """Stride-2 conv with asymmetric (right/bottom) padding, halving H and W."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    """Nearest-neighbour 2x upsample followed by a 3x3 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))
