"""Train the BPE tokenizer and pre-encode the dataset into VAE latents.

Usage: python scripts/prepare_data.py --config configs/t4_128.yaml --vae runs/vae/vae.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sd.config import Config, load_config
from sd.data import build_records, cache_latents
from sd.tokenizer import TextTokenizer
from sd.utils import get_device, load_checkpoint, seed_everything
from sd.vae import AutoencoderKL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--vae", required=True, help="trained VAE checkpoint (runs/vae/vae.pt)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    device = get_device()
    records = build_records(cfg.data)
    print(f"{len(records)} training images")

    tokenizer = TextTokenizer.train(records.captions(), cfg.text.vocab_size, cfg.text.max_length)
    tokenizer.save(cfg.data.tokenizer_path)
    print(f"Tokenizer: {tokenizer.vocab_size} tokens -> {cfg.data.tokenizer_path}")
    if tokenizer.vocab_size > cfg.text.vocab_size:
        raise SystemExit("tokenizer vocabulary exceeds text.vocab_size")

    blob = load_checkpoint(args.vae)
    vae = AutoencoderKL(Config.from_dict(blob["config"]).vae).to(device)
    vae.load_state_dict(blob["model"])
    scale = cache_latents(vae, records, cfg.data.image_size, cfg.data.cache_path, args.batch_size, device)
    print(f"Cached latents -> {cfg.data.cache_path} (scale factor {scale:.5f})")


if __name__ == "__main__":
    main()
