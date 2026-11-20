"""Train the text-conditioned latent diffusion model on cached latents.

Usage: python scripts/train_ldm.py --config configs/t4_128.yaml --vae runs/vae/vae.pt
Re-running the same command resumes from <out_dir>/latest.pt, so Colab disconnects only cost the last interval.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from sd.config import Config, load_config
from sd.data import LatentCaptionDataset
from sd.pipeline import LatentDiffusionPipeline
from sd.schedulers import NoiseSchedule
from sd.text import TextEncoder
from sd.tokenizer import TextTokenizer
from sd.trainer import LDMTrainer
from sd.unet import UNet
from sd.utils import count_parameters, get_device, load_checkpoint, make_grid, seed_everything
from sd.vae import AutoencoderKL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    seed_everything(cfg.train.seed)
    device = get_device()

    dataset = LatentCaptionDataset(cfg.data.cache_path)
    tokenizer = TextTokenizer.load(cfg.data.tokenizer_path, cfg.text.max_length)
    schedule = NoiseSchedule(cfg.diffusion)
    unet, text = UNet(cfg.unet), TextEncoder(cfg.text)
    print(f"UNet {count_parameters(unet) / 1e6:.1f}M | text encoder {count_parameters(text) / 1e6:.1f}M | {len(dataset)} samples")

    blob = load_checkpoint(args.vae)
    vae = AutoencoderKL(Config.from_dict(blob["config"]).vae).to(device).eval()
    vae.load_state_dict(blob["model"])
    vae.scale_factor.fill_(dataset.scale_factor)

    sample_dir = os.path.join(cfg.train.out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    def log_samples(trainer: LDMTrainer) -> None:
        """Render the fixed prompts with the EMA weights (the trainer has already swapped them in)."""
        pipe = LatentDiffusionPipeline(vae, trainer.text_encoder, tokenizer, trainer.unet, schedule, cfg.data.image_size, device)
        images = pipe(list(cfg.train.sample_prompts), num_inference_steps=25, guidance_scale=5.0, seed=0)
        make_grid(images).save(os.path.join(sample_dir, f"step_{trainer.step:06d}.png"))

    trainer = LDMTrainer(
        cfg, unet, text, tokenizer, schedule, dataset, device,
        scale_factor=dataset.scale_factor, on_sample=log_samples,
    )
    if trainer.maybe_resume():
        print(f"Resumed from step {trainer.step}")
    trainer.fit()
    print(f"Done. Checkpoint: {os.path.join(cfg.train.out_dir, 'latest.pt')}")


if __name__ == "__main__":
    main()
