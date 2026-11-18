"""Datasets: raw (image, caption) records and, later, cached VAE latents."""
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import DataConfig
from .utils import fit_image, pil_to_tensor
from torch.utils.data import DataLoader

from .utils import load_checkpoint, save_checkpoint

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class HFRecords:
    """Adapter exposing a HuggingFace `datasets.Dataset` as (PIL image, caption) pairs."""

    def __init__(self, dataset, image_column: str = "image", text_column: str = "text"):
        self.dataset, self.image_column, self.text_column = dataset, image_column, text_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        row = self.dataset[idx]
        caption = row[self.text_column]
        if isinstance(caption, (list, tuple)):  # several captions per image: pick one
            caption = random.choice(caption)
        return row[self.image_column].convert("RGB"), str(caption)

    def captions(self) -> List[str]:
        col = self.dataset[self.text_column]
        return [c if isinstance(c, str) else c[0] for c in col]


class FolderRecords:
    """A directory of images; `name.txt` next to `name.png` holds the caption.

    Falls back to the file name (underscores/dashes -> spaces) when no caption file exists.
    """

    def __init__(self, folder: str):
        root = Path(folder)
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {folder}")

    def __len__(self) -> int:
        return len(self.paths)

    def _caption(self, path: Path) -> str:
        txt = path.with_suffix(".txt")
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
        return path.stem.replace("_", " ").replace("-", " ")

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        path = self.paths[idx]
        return Image.open(path).convert("RGB"), self._caption(path)

    def captions(self) -> List[str]:
        return [self._caption(p) for p in self.paths]


def build_records(cfg: DataConfig):
    """Records for the configured source: a local folder if set, otherwise a HF dataset."""
    if cfg.folder:
        return FolderRecords(cfg.folder)
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, split=cfg.split)
    return HFRecords(ds, cfg.image_column, cfg.text_column)


class ImageCaptionDataset(Dataset):
    """Resize + center-crop images to `image_size`, scaled to [-1, 1], with optional h-flip."""

    def __init__(self, records, image_size: int = 128, flip: bool = True):
        self.records, self.image_size, self.flip = records, image_size, flip

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        image, caption = self.records[idx]
        x = pil_to_tensor(fit_image(image, self.image_size))
        if self.flip and random.random() < 0.5:
            x = torch.flip(x, dims=[-1])
        return {"image": x, "caption": caption}


@torch.no_grad()
def cache_latents(
    vae,
    records,
    image_size: int,
    out_path: str,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
) -> float:
    """Encode every image (and its mirror) once with the frozen VAE and store the posterior moments.

    Training the diffusion model on cached latents skips the VAE encoder entirely, which is the
    single biggest speed-up on a T4. Also estimates the latent scale factor (1 / std of the latents)
    so the diffusion model sees roughly unit-variance inputs. Returns that scale factor.
    """
    device = device or next(vae.parameters()).device
    vae.eval()
    loader = DataLoader(
        ImageCaptionDataset(records, image_size, flip=False), batch_size=batch_size, num_workers=num_workers
    )
    moments, captions = [], []
    for batch in loader:
        x = batch["image"].to(device)
        both = torch.stack([vae.encode_moments(x), vae.encode_moments(torch.flip(x, dims=[-1]))], dim=1)
        moments.append(both.half().cpu())
        captions.extend(batch["caption"])
    moments = torch.cat(moments)  # (N, 2, 2*C, h, w)

    mean, logvar = moments[:, 0].float().chunk(2, dim=1)
    z = mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * torch.randn_like(mean)
    scale_factor = float(1.0 / z.std())
    vae.scale_factor.fill_(scale_factor)

    save_checkpoint(
        {"moments": moments, "captions": captions, "scale_factor": scale_factor, "image_size": image_size},
        out_path,
    )
    return scale_factor


class LatentCaptionDataset(Dataset):
    """Cached latents + captions. Each access samples z ~ q(z|x) and picks the mirrored copy at random."""

    def __init__(self, path: str):
        blob = load_checkpoint(path)
        self.moments: torch.Tensor = blob["moments"]
        self.captions: List[str] = blob["captions"]
        self.scale_factor: float = blob["scale_factor"]
        self.image_size: int = blob["image_size"]

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> dict:
        moments = self.moments[idx, random.randint(0, 1)].float()
        mean, logvar = moments.chunk(2, dim=0)
        z = mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * torch.randn_like(mean)
        return {"latent": z * self.scale_factor, "caption": self.captions[idx]}
