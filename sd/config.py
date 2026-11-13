"""Typed configuration for every component of the latent diffusion model."""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass
class VAEConfig:
    in_channels: int = 3
    latent_channels: int = 4
    base_channels: int = 64
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)  # 3 downsamples -> 8x spatial reduction
    num_res_blocks: int = 2
    groups: int = 32
    dropout: float = 0.0
    scale_factor: float = 0.18215  # re-estimated from data when latents are cached
    kl_weight: float = 1e-6
    perceptual_weight: float = 0.5


@dataclass
class UNetConfig:
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    attention_levels: Tuple[int, ...] = (0, 1, 2)
    head_dim: int = 32
    transformer_depth: int = 1
    context_dim: int = 256
    dropout: float = 0.0
    groups: int = 32
    gradient_checkpointing: bool = False


@dataclass
class TextConfig:
    vocab_size: int = 8192
    max_length: int = 77
    width: int = 256
    layers: int = 6
    heads: int = 4


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"  # linear | scaled_linear | cosine
    beta_start: float = 0.00085
    beta_end: float = 0.012
    prediction_type: str = "epsilon"  # epsilon | v_prediction
    zero_terminal_snr: bool = False  # requires v_prediction


@dataclass
class DataConfig:
    dataset: str = "lambdalabs/pokemon-blip-captions"
    split: str = "train"
    image_column: str = "image"
    text_column: str = "text"
    folder: Optional[str] = None  # use a local folder of images (+ .txt captions) instead
    image_size: int = 128
    cache_path: str = "cache/latents.pt"
    tokenizer_path: str = "cache/tokenizer.json"


@dataclass
class VAETrainConfig:
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 8000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    mixed_precision: bool = True
    log_every: int = 50
    ckpt_every: int = 1000
    out_dir: str = "runs/vae"
    seed: int = 42
    num_workers: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 1e-4
    text_lr_mult: float = 0.5
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    max_steps: int = 20000
    grad_clip: float = 1.0
    ema_decay: float = 0.9995  # 0 disables EMA
    cond_drop_prob: float = 0.1  # caption dropout, enables classifier-free guidance
    min_snr_gamma: float = 5.0  # 0 disables Min-SNR loss weighting
    mixed_precision: bool = True
    train_text_encoder: bool = True
    log_every: int = 50
    ckpt_every: int = 1000
    sample_every: int = 1000
    out_dir: str = "runs/ldm"
    seed: int = 42
    num_workers: int = 0
    sample_prompts: Tuple[str, ...] = (
        "a cartoon creature with big eyes",
        "a blue water type monster",
        "a small yellow animal with red cheeks",
        "a green dragon with wings",
    )


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.0
    targets: Tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out")
    lr: float = 1e-4
    max_steps: int = 1500
    out_dir: str = "runs/lora"


@dataclass
class Config:
    vae: VAEConfig = field(default_factory=VAEConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    text: TextConfig = field(default_factory=TextConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vae_train: VAETrainConfig = field(default_factory=VAETrainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return _from_dict(cls, data)


def _from_dict(cls, data: Dict[str, Any]):
    """Recursively build a (nested) dataclass from a plain dict, rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name in known & set(data):
        value, hint = data[name], hints[name]
        if dataclasses.is_dataclass(hint):
            value = _from_dict(hint, value)
        elif typing.get_origin(hint) is tuple:
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Optional[str] = None) -> Config:
    """Load a YAML config; missing keys fall back to the defaults above."""
    if path is None:
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        return Config.from_dict(yaml.safe_load(f) or {})


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
