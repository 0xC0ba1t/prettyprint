"""Fine-tune a trained model on a small folder of images with LoRA adapters.

Usage:
  python scripts/train_lora.py --ldm runs/ldm/latest.pt --vae runs/vae/vae.pt \
      --tokenizer cache/tokenizer.json --folder my_images/

Captions come from `name.txt` next to each image (or the file name). Only the adapters are trained and saved
(a few MB), so one base model can serve many styles.
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sd.config import Config
from sd.data import FolderRecords, LatentCaptionDataset, cache_latents
from sd.lora import inject_lora, lora_parameters, save_lora
from sd.pipeline import LatentDiffusionPipeline
from sd.trainer import LDMTrainer
from sd.utils import get_device, load_checkpoint, make_grid, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--cache", default="cache/lora_latents.pt")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    device = get_device()
    base_cfg = Config.from_dict(load_checkpoint(args.ldm)["config"])
    lc = base_cfg.lora
    rank = args.rank or lc.rank
    seed_everything(base_cfg.train.seed)

    pipe = LatentDiffusionPipeline.from_checkpoints(args.ldm, args.vae, args.tokenizer, device)
    cache_latents(pipe.vae, FolderRecords(args.folder), base_cfg.data.image_size, args.cache, device=device)
    dataset = LatentCaptionDataset(args.cache)

    added = inject_lora(pipe.unet, rank, lc.alpha * rank / lc.rank, lc.targets, lc.dropout)
    print(f"Injected {added} LoRA layers (rank {rank}); {sum(p.numel() for p in lora_parameters(pipe.unet)) / 1e6:.2f}M trainable")

    out_dir = args.out_dir or lc.out_dir
    train_cfg = dataclasses.replace(
        base_cfg.train, lr=lc.lr, max_steps=args.max_steps or lc.max_steps, warmup_steps=50,
        out_dir=out_dir, ckpt_every=250, sample_every=250, batch_size=min(base_cfg.train.batch_size, 8),
        ema_decay=0.0, train_text_encoder=False,
    )
    cfg = dataclasses.replace(base_cfg, train=train_cfg)

    def log_samples(trainer: LDMTrainer) -> None:
        images = pipe(list(cfg.train.sample_prompts), num_inference_steps=25, seed=0)
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        make_grid(images).save(os.path.join(out_dir, "samples", f"step_{trainer.step:05d}.png"))

    trainer = LDMTrainer(
        cfg, pipe.unet, pipe.text_encoder, pipe.tokenizer, pipe.schedule, dataset, device,
        lora_only=True, scale_factor=dataset.scale_factor, on_sample=log_samples,
    )
    trainer.maybe_resume()
    trainer.fit()
    path = os.path.join(out_dir, "lora.pt")
    save_lora(pipe.unet, path, rank, lc.alpha * rank / lc.rank, lc.targets)
    print(f"Saved adapter to {path}")


if __name__ == "__main__":
    main()
