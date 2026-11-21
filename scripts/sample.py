"""Generate images: text-to-image, img2img (--init) or inpainting (--init + --mask), optionally with a LoRA.

Examples:
  python scripts/sample.py --ldm runs/ldm/latest.pt --vae runs/vae/vae.pt --tokenizer cache/tokenizer.json \
      --prompt "a blue water monster" --num 4 --out outputs/blue.png
  python scripts/sample.py ... --prompt "a red dragon" --init in.png --strength 0.6
  python scripts/sample.py ... --prompt "a hat" --init in.png --mask mask.png      # white = repaint
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from sd.lora import load_lora
from sd.pipeline import LatentDiffusionPipeline
from sd.utils import get_device, make_grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ldm", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="")
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--sampler", default="dpmpp", choices=["ddpm", "ddim", "dpmpp"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--init", default=None, help="input image for img2img / inpainting")
    p.add_argument("--mask", default=None, help="inpainting mask (white = regenerate)")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--lora", default=None, help="LoRA adapter file")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--out", default="outputs/sample.png")
    args = p.parse_args()

    pipe = LatentDiffusionPipeline.from_checkpoints(args.ldm, args.vae, args.tokenizer, get_device(), use_ema=not args.no_ema)
    if args.lora:
        load_lora(pipe.unet, args.lora, scale=args.lora_scale, merge=True)

    common = dict(
        negative_prompt=args.negative, num_images=args.num, num_inference_steps=args.steps,
        guidance_scale=args.guidance, sampler=args.sampler, seed=args.seed,
    )
    if args.init and args.mask:
        images = pipe.inpaint(
            args.prompt, Image.open(args.init), Image.open(args.mask),
            strength=1.0 if args.strength is None else args.strength, **common,
        )
    elif args.init:
        images = pipe.img2img(
            args.prompt, Image.open(args.init), strength=0.7 if args.strength is None else args.strength, **common
        )
    else:
        images = pipe(args.prompt, **common)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make_grid(images).save(args.out)
    print(f"Wrote {len(images)} image(s) to {args.out}")


if __name__ == "__main__":
    main()
