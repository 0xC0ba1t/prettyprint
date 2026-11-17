"""Text-conditioned UNet that predicts noise (or velocity) in latent space."""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SpatialTransformer
from .blocks import Downsample, ResnetBlock, Upsample, group_norm
from .config import UNetConfig
from .embeddings import TimestepEmbedder
from torch.utils.checkpoint import checkpoint


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model_channels
        temb_dim = mc * 4
        self.time_embed = TimestepEmbedder(mc, temb_dim)
        self.conv_in = nn.Conv2d(cfg.in_channels, mc, 3, padding=1)

        def attn(ch: int) -> SpatialTransformer:
            return SpatialTransformer(ch, cfg.head_dim, cfg.context_dim, cfg.transformer_depth, cfg.groups, cfg.dropout)

        def res(cin: int, cout: int) -> ResnetBlock:
            return ResnetBlock(cin, cout, temb_dim, cfg.dropout, cfg.groups)

        # --- encoder: every block is a ModuleList of layers; channel counts of the skips are recorded
        ch = mc
        skip_channels = [ch]
        self.down_blocks = nn.ModuleList()
        for level, mult in enumerate(cfg.ch_mult):
            out_ch = mc * mult
            for _ in range(cfg.num_res_blocks):
                layers: List[nn.Module] = [res(ch, out_ch)]
                ch = out_ch
                if level in cfg.attention_levels:
                    layers.append(attn(ch))
                self.down_blocks.append(nn.ModuleList(layers))
                skip_channels.append(ch)
            if level != len(cfg.ch_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)

        self.mid_block = nn.ModuleList([res(ch, ch), attn(ch), res(ch, ch)])

        # --- decoder: consumes one skip per block (hence num_res_blocks + 1 blocks per level)
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(cfg.ch_mult))):
            out_ch = mc * mult
            for i in range(cfg.num_res_blocks + 1):
                layers = [res(ch + skip_channels.pop(), out_ch)]
                ch = out_ch
                if level in cfg.attention_levels:
                    layers.append(attn(ch))
                if level != 0 and i == cfg.num_res_blocks:
                    layers.append(Upsample(ch))
                self.up_blocks.append(nn.ModuleList(layers))

        self.norm_out = group_norm(ch, cfg.groups)
        self.conv_out = nn.Conv2d(ch, cfg.out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # predict zeros at init
        nn.init.zeros_(self.conv_out.bias)
        self.gradient_checkpointing = cfg.gradient_checkpointing

    @staticmethod
    def _apply_block(block: nn.ModuleList, h: torch.Tensor, temb: torch.Tensor, context: Optional[torch.Tensor]):
        for layer in block:
            if isinstance(layer, ResnetBlock):
                h = layer(h, temb)
            elif isinstance(layer, SpatialTransformer):
                h = layer(h, context)
            else:
                h = layer(h)
        return h

    def _run(self, block: nn.ModuleList, h: torch.Tensor, temb: torch.Tensor, context: Optional[torch.Tensor]):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self._apply_block, block, h, temb, context, use_reentrant=False)
        return self._apply_block(block, h, temb, context)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, C, H, W) noisy latents; timesteps: (B,) ints; context: (B, L, context_dim)."""
        temb = self.time_embed(timesteps)
        h = self.conv_in(x)
        skips = [h]
        for block in self.down_blocks:
            h = self._run(block, h, temb, context)
            skips.append(h)
        h = self._run(self.mid_block, h, temb, context)
        for block in self.up_blocks:
            h = self._run(block, torch.cat([h, skips.pop()], dim=1), temb, context)
        return self.conv_out(F.silu(self.norm_out(h)))
