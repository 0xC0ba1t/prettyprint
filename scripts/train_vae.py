"""Train the KL autoencoder.  Usage: python scripts/train_vae.py --config configs/t4_128.yaml"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sd.config import load_config
from sd.data import ImageCaptionDataset, build_records
from sd.losses import VAELoss, VGGPerceptualLoss
from sd.utils import (
    cosine_warmup,
    count_parameters,
    get_device,
    load_checkpoint,
    make_grid,
    save_checkpoint,
    seed_everything,
    tensor_to_pil,
)
from sd.vae import AutoencoderKL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-perceptual", action="store_true", help="skip the VGG loss (needs no torchvision)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tc = cfg.vae_train
    max_steps = args.max_steps or tc.max_steps
    seed_everything(tc.seed)
    device = get_device()
    os.makedirs(tc.out_dir, exist_ok=True)

    vae = AutoencoderKL(cfg.vae).to(device)
    print(f"VAE parameters: {count_parameters(vae) / 1e6:.1f}M on {device}")
    perceptual = None if args.no_perceptual else VGGPerceptualLoss().to(device)
    criterion = VAELoss(cfg.vae, perceptual)

    dataset = ImageCaptionDataset(build_records(cfg.data), cfg.data.image_size, flip=True)
    loader = DataLoader(
        dataset, batch_size=tc.batch_size, shuffle=True, drop_last=len(dataset) >= tc.batch_size,
        num_workers=tc.num_workers, pin_memory=device.type == "cuda", persistent_workers=tc.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(vae.parameters(), lr=tc.lr, betas=(0.9, 0.99), weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_warmup(s, tc.warmup_steps, max_steps, 0.1))
    amp = tc.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    step, latest = 0, os.path.join(tc.out_dir, "latest.pt")
    if os.path.exists(latest) and not args.no_resume:
        ckpt = load_checkpoint(latest, device)
        vae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        print(f"Resumed from step {step}")

    def save(path: str, full: bool) -> None:
        blob = {"model": vae.state_dict(), "config": cfg.to_dict(), "step": step}
        if full:
            blob.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
        save_checkpoint(blob, path)

    vae.train()
    pbar = tqdm(total=max_steps, initial=step, desc="vae", dynamic_ncols=True)
    while step < max_steps:
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                recon, posterior = vae(x)
            loss, logs = criterion(x, recon, posterior)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), tc.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            pbar.update(1)
            if step % tc.log_every == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})
            if step % tc.ckpt_every == 0 or step == max_steps:
                save(latest, full=True)
                n = min(4, x.shape[0])
                pairs = [tensor_to_pil(t) for t in torch.cat([x[:n], recon[:n].float()])]
                make_grid(pairs, cols=n).save(os.path.join(tc.out_dir, f"recon_{step:06d}.png"))
            if step >= max_steps:
                break
    pbar.close()
    save(os.path.join(tc.out_dir, "vae.pt"), full=False)
    print(f"Saved {os.path.join(tc.out_dir, 'vae.pt')}")


if __name__ == "__main__":
    main()
