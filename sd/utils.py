"""Small shared helpers: seeding, devices, checkpoints, image conversion."""
import math
import os
import random
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def save_checkpoint(obj: dict, path: str) -> None:
    """Atomic save: a Colab disconnect mid-write must never corrupt the latest checkpoint."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def cosine_warmup(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.0) -> float:
    """LR multiplier: linear warmup followed by cosine decay to `min_ratio`."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB image -> float tensor (3, H, W) in [-1, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Float tensor (3, H, W) in [-1, 1] -> PIL RGB image."""
    x = ((x.detach().float().clamp(-1, 1) + 1.0) * 127.5).round().byte().cpu()
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def fit_image(img: Image.Image, size: int, resample=Image.BICUBIC) -> Image.Image:
    """Resize the short side to `size` and center-crop to a square."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), resample)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def make_grid(images: Sequence[Image.Image], cols: Optional[int] = None) -> Image.Image:
    cols = cols or min(len(images), 4)
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid
