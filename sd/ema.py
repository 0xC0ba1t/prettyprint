"""Exponential moving average of model weights (used for sampling and the final checkpoint)."""
from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """Keeps a float32 shadow copy of every trainable parameter.

    The decay is warmed up as min(decay, (1 + n) / (10 + n)) so the average is useful early in training.
    """

    def __init__(self, module: nn.Module, decay: float = 0.9995, warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone().float() for name, p in module.named_parameters() if p.requires_grad
        }

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        self.num_updates += 1
        weight = 1.0 - self.current_decay()
        for name, p in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(p.detach().float(), weight)

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name].to(p.dtype))

    @contextmanager
    def average_parameters(self, module: nn.Module):
        """Temporarily load the averaged weights into `module`, restoring the originals on exit."""
        backup = {name: p.detach().clone() for name, p in module.named_parameters() if name in self.shadow}
        self.copy_to(module)
        try:
            yield module
        finally:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if name in backup:
                        p.copy_(backup[name])

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        self.num_updates = state["num_updates"]
        self.decay = state.get("decay", self.decay)
