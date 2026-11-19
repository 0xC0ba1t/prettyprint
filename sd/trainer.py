"""Training loop for the latent diffusion model (UNet + text encoder on cached latents)."""
import json
import os
import random
from contextlib import nullcontext
from typing import Callable, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import Config
from .ema import EMA
from .schedulers import NoiseSchedule
from .tokenizer import TextTokenizer
from .utils import cosine_warmup, save_checkpoint
from .utils import load_checkpoint


class LDMTrainer:
    """Trains `unet` (and optionally `text_encoder`) to denoise cached VAE latents.

    Memory/speed choices for a 16 GB T4: fp16 autocast with a GradScaler (the T4 has no bf16),
    pre-encoded latents (the VAE never runs during training), gradient accumulation, optional
    gradient checkpointing inside the UNet, and EMA weights for sampling.
    """

    def __init__(
        self,
        cfg: Config,
        unet: nn.Module,
        text_encoder: nn.Module,
        tokenizer: TextTokenizer,
        schedule: NoiseSchedule,
        dataset: Dataset,
        device: torch.device,
        lora_only: bool = False,
        scale_factor: float = 0.18215,
        on_sample: Optional[Callable[["LDMTrainer"], None]] = None,
    ):
        self.cfg, self.tc = cfg, cfg.train
        self.device = device
        self.unet, self.text_encoder = unet.to(device), text_encoder.to(device)
        self.tokenizer, self.schedule = tokenizer, schedule
        self.scale_factor = scale_factor
        self.lora_only = lora_only
        self.on_sample = on_sample
        self.modules = nn.ModuleDict({"unet": self.unet, "text": self.text_encoder})

        if lora_only:  # adapters only: everything else stays frozen
            for name, p in self.modules.named_parameters():
                p.requires_grad_("lora_" in name)
        elif not self.tc.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: cosine_warmup(s, self.tc.warmup_steps, self.tc.max_steps, 0.05)
        )
        self.amp = self.tc.mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        use_ema = self.tc.ema_decay > 0 and not lora_only
        self.ema = EMA(self.modules, self.tc.ema_decay) if use_ema else None
        self.step = 0
        self.history: List[float] = []

        batch = min(self.tc.batch_size, len(dataset))
        self.loader = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=True,
            drop_last=len(dataset) >= self.tc.batch_size,
            num_workers=self.tc.num_workers,
            pin_memory=device.type == "cuda",
        )
        os.makedirs(self.tc.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ setup
    def _build_optimizer(self) -> torch.optim.Optimizer:
        groups = []
        for module, lr in ((self.unet, self.tc.lr), (self.text_encoder, self.tc.lr * self.tc.text_lr_mult)):
            params = [p for p in module.parameters() if p.requires_grad]
            decay = [p for p in params if p.ndim > 1]
            no_decay = [p for p in params if p.ndim <= 1]  # norms and biases
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": self.tc.weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    def _trainable_params(self) -> List[nn.Parameter]:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def _batches(self) -> Iterator[Dict]:
        while True:
            for batch in self.loader:
                yield batch

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.amp else nullcontext()

    def set_train(self, train: bool) -> None:
        self.unet.train(train)
        self.text_encoder.train(train and self.tc.train_text_encoder and not self.lora_only)

    # ------------------------------------------------------------------ one optimisation step
    def loss_on_batch(self, batch: Dict) -> torch.Tensor:
        z = batch["latent"].to(self.device)
        captions = list(batch["caption"])
        # Caption dropout teaches the model an unconditional distribution for classifier-free guidance.
        captions = ["" if random.random() < self.tc.cond_drop_prob else c for c in captions]
        ids = self.tokenizer(captions).to(self.device)
        noise = torch.randn_like(z)
        t = torch.randint(0, self.schedule.num_timesteps, (z.shape[0],), device=self.device)
        z_t = self.schedule.add_noise(z, noise, t)
        target = self.schedule.training_target(z, noise, t)
        with self._autocast():
            pred = self.unet(z_t, t, self.text_encoder(ids))
        loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
        gamma = self.tc.min_snr_gamma
        if gamma > 0:  # Min-SNR-gamma weighting (Hang et al. 2023): faster, better-balanced convergence
            snr = self.schedule.snr(t)
            if self.schedule.prediction_type == "epsilon":
                weight = torch.clamp(snr, max=gamma) / snr
            else:
                weight = torch.clamp(snr, max=gamma) / (snr + 1.0)
            loss = loss * weight
        return loss.mean()

    def train_step(self, batches: Iterator[Dict]) -> float:
        total = 0.0
        for _ in range(self.tc.grad_accum):
            loss = self.loss_on_batch(next(batches)) / self.tc.grad_accum
            self.scaler.scale(loss).backward()
            total += loss.item()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self._trainable_params(), self.tc.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        if self.ema is not None:
            self.ema.update(self.modules)
        self.step += 1
        self.history.append(total)
        return total

    # ------------------------------------------------------------------ main loop
    def ema_scope(self):
        return self.ema.average_parameters(self.modules) if self.ema is not None else nullcontext()

    def fit(self, max_steps: Optional[int] = None) -> None:
        max_steps = max_steps or self.tc.max_steps
        self.set_train(True)
        batches = self._batches()
        log_path = os.path.join(self.tc.out_dir, "log.jsonl")
        pbar = tqdm(total=max_steps, initial=self.step, desc="train", dynamic_ncols=True)
        while self.step < max_steps:
            loss = self.train_step(batches)
            pbar.update(1)
            if self.step % self.tc.log_every == 0:
                recent = sum(self.history[-self.tc.log_every :]) / min(len(self.history), self.tc.log_every)
                pbar.set_postfix(loss=f"{recent:.4f}", lr=f"{self.lr_scheduler.get_last_lr()[0]:.2e}")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": self.step, "loss": recent, "lr": self.lr_scheduler.get_last_lr()[0]}) + "\n")
            if self.step % self.tc.sample_every == 0 and self.on_sample is not None:
                self.set_train(False)
                with self.ema_scope():
                    self.on_sample(self)
                self.set_train(True)
            if self.step % self.tc.ckpt_every == 0 or self.step == max_steps:
                self.save()
        pbar.close()

    # ------------------------------------------------------------------ checkpointing (Colab-safe)
    def _model_state(self) -> Dict[str, torch.Tensor]:
        state = self.modules.state_dict()
        if self.lora_only:
            state = {k: v for k, v in state.items() if "lora_" in k}
        return state

    def save(self, name: str = "latest.pt") -> str:
        path = os.path.join(self.tc.out_dir, name)
        save_checkpoint(
            {
                "step": self.step,
                "model": self._model_state(),
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": self.cfg.to_dict(),
                "scale_factor": self.scale_factor,
            },
            path,
        )
        return path

    def load(self, path: str) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        self.modules.load_state_dict(ckpt["model"], strict=not self.lora_only)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema is not None and ckpt.get("ema"):
            self.ema.load_state_dict(ckpt["ema"])
        self.step = ckpt["step"]

    def maybe_resume(self) -> bool:
        """Pick up from <out_dir>/latest.pt if it exists (so a disconnected Colab session just continues)."""
        path = os.path.join(self.tc.out_dir, "latest.pt")
        if os.path.exists(path):
            self.load(path)
            return True
        return False
