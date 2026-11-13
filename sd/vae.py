"""KL-regularised convolutional autoencoder (the 'VAE' of latent diffusion)."""
import torch
import torch.nn as nn

from .attention import VAEAttnBlock
from .blocks import Downsample, ResnetBlock, Upsample, group_norm
from .config import VAEConfig


class Encoder(nn.Module):
    """image (B,3,H,W) -> moments (B, 2*latent, H/8, W/8)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        ch = cfg.base_channels
        layers = [nn.Conv2d(cfg.in_channels, ch, 3, padding=1)]
        in_ch = ch
        for level, mult in enumerate(cfg.ch_mult):
            out_ch = ch * mult
            for _ in range(cfg.num_res_blocks):
                layers.append(ResnetBlock(in_ch, out_ch, dropout=cfg.dropout, groups=cfg.groups))
                in_ch = out_ch
            if level != len(cfg.ch_mult) - 1:
                layers.append(Downsample(in_ch))
        layers += [
            ResnetBlock(in_ch, in_ch, dropout=cfg.dropout, groups=cfg.groups),
            VAEAttnBlock(in_ch, cfg.groups),
            ResnetBlock(in_ch, in_ch, dropout=cfg.dropout, groups=cfg.groups),
            group_norm(in_ch, cfg.groups),
            nn.SiLU(),
            nn.Conv2d(in_ch, 2 * cfg.latent_channels, 3, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
