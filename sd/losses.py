"""Training losses for the autoencoder: L1 + VGG perceptual + KL."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VAEConfig
from .vae import DiagonalGaussian


class VGGPerceptualLoss(nn.Module):
    """LPIPS-style loss using frozen VGG16 features (unit-normalised per location, no learned heads).

    torchvision is imported lazily so the rest of the package works without it.
    """

    # indices of relu1_2, relu2_2, relu3_3, relu4_3 inside vgg16().features
    _CUTS = (4, 9, 16, 23)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg16(weights=weights).features[: self._CUTS[-1]].eval()
        for p in features.parameters():
            p.requires_grad_(False)
        start, slices = 0, []
        for cut in self._CUTS:
            slices.append(nn.Sequential(*[features[i] for i in range(start, cut)]))
            start = cut
        self.slices = nn.ModuleList(slices)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # keep the frozen backbone in eval mode
        return super().train(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = ((x.float() + 1) / 2 - self.mean) / self.std
        y = ((y.float() + 1) / 2 - self.mean) / self.std
        loss = x.new_zeros(())
        for block in self.slices:
            x, y = block(x), block(y)
            nx = x / (x.norm(dim=1, keepdim=True) + 1e-10)
            ny = y / (y.norm(dim=1, keepdim=True) + 1e-10)
            loss = loss + ((nx - ny) ** 2).sum(dim=1).mean()
        return loss


class VAELoss(nn.Module):
    def __init__(self, cfg: VAEConfig, perceptual: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.perceptual = perceptual

    def forward(
        self, x: torch.Tensor, recon: torch.Tensor, posterior: DiagonalGaussian
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1 = F.l1_loss(recon.float(), x.float())
        kl = posterior.kl().mean()
        total = l1 + self.cfg.kl_weight * kl
        logs = {"l1": l1.item(), "kl": kl.item()}
        if self.perceptual is not None and self.cfg.perceptual_weight > 0:
            perc = self.perceptual(recon, x)
            total = total + self.cfg.perceptual_weight * perc
            logs["perceptual"] = perc.item()
        logs["total"] = total.item()
        return total, logs
