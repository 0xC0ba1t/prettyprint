"""Tiny models shared by the pipeline and trainer tests (CPU-friendly)."""
import torch

from sd.config import Config, DiffusionConfig, TextConfig, UNetConfig, VAEConfig
from sd.pipeline import LatentDiffusionPipeline
from sd.schedulers import NoiseSchedule
from sd.text import TextEncoder
from sd.tokenizer import TextTokenizer
from sd.unet import UNet
from sd.vae import AutoencoderKL

CAPTIONS = ["a red square", "a blue circle", "a green triangle", "a yellow star"] * 4


def tiny_config(**diffusion) -> Config:
    cfg = Config()
    cfg.vae = VAEConfig(base_channels=16, ch_mult=(1, 2, 2, 2), num_res_blocks=1, groups=8)
    cfg.unet = UNetConfig(model_channels=32, ch_mult=(1, 2), num_res_blocks=1, attention_levels=(1,), head_dim=16,
                          context_dim=32, groups=8)
    cfg.text = TextConfig(vocab_size=300, max_length=12, width=32, layers=2, heads=4)
    cfg.diffusion = DiffusionConfig(**diffusion)
    cfg.data.image_size = 32
    return cfg


def tiny_pipeline(**diffusion) -> LatentDiffusionPipeline:
    torch.manual_seed(0)
    cfg = tiny_config(**diffusion)
    tokenizer = TextTokenizer.train(CAPTIONS, vocab_size=300, max_length=cfg.text.max_length)
    unet = UNet(cfg.unet)
    torch.nn.init.normal_(unet.conv_out.weight, std=0.02)  # non-trivial denoiser
    return LatentDiffusionPipeline(
        AutoencoderKL(cfg.vae), TextEncoder(cfg.text), tokenizer, unet, NoiseSchedule(cfg.diffusion),
        image_size=cfg.data.image_size, device=torch.device("cpu"),
    )
