"""KL-regularised convolutional autoencoder (the 'VAE' of latent diffusion)."""
import torch
import torch.nn as nn

from .attention import VAEAttnBlock
from .blocks import Downsample, ResnetBlock, Upsample, group_norm
from .config import VAEConfig
from typing import Optional


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


class Decoder(nn.Module):
    """latent (B, latent, H/8, W/8) -> image (B,3,H,W)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        ch = cfg.base_channels
        in_ch = ch * cfg.ch_mult[-1]
        layers = [
            nn.Conv2d(cfg.latent_channels, in_ch, 3, padding=1),
            ResnetBlock(in_ch, in_ch, dropout=cfg.dropout, groups=cfg.groups),
            VAEAttnBlock(in_ch, cfg.groups),
            ResnetBlock(in_ch, in_ch, dropout=cfg.dropout, groups=cfg.groups),
        ]
        for level, mult in reversed(list(enumerate(cfg.ch_mult))):
            out_ch = ch * mult
            for _ in range(cfg.num_res_blocks + 1):
                layers.append(ResnetBlock(in_ch, out_ch, dropout=cfg.dropout, groups=cfg.groups))
                in_ch = out_ch
            if level != 0:
                layers.append(Upsample(in_ch))
        layers += [group_norm(in_ch, cfg.groups), nn.SiLU(), nn.Conv2d(in_ch, cfg.in_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DiagonalGaussian:
    """Posterior q(z|x) parameterised by concatenated mean / log-variance channels."""

    def __init__(self, moments: torch.Tensor):
        self.mean, logvar = moments.float().chunk(2, dim=1)
        self.logvar = logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, generator=generator, dtype=self.mean.dtype)
        return self.mean + self.std * noise.to(self.mean.device)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """KL(q || N(0, I)) summed over latent dims -> shape (B,)."""
        return 0.5 * torch.sum(self.mean**2 + self.logvar.exp() - 1.0 - self.logvar, dim=[1, 2, 3])


class AutoencoderKL(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.quant_conv = nn.Conv2d(2 * cfg.latent_channels, 2 * cfg.latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1)
        self.register_buffer("scale_factor", torch.tensor(float(cfg.scale_factor)))

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.cfg.ch_mult) - 1)

    def encode_moments(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(x))

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(self.encode_moments(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior

    # Convenience wrappers working in the *scaled* latent space the diffusion model sees.
    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor, sample: bool = True, generator: Optional[torch.Generator] = None):
        posterior = self.encode(x)
        z = posterior.sample(generator) if sample else posterior.mode()
        return z * self.scale_factor

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z / self.scale_factor)
