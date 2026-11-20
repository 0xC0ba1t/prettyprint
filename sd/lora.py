"""Low-rank adapters (LoRA) for the UNet's attention projections."""
import math
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_checkpoint, save_checkpoint

DEFAULT_TARGETS = ("to_q", "to_k", "to_v", "to_out")


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * multiplier * B(A(x)), with W frozen. B starts at zero (identity at init)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.multiplier = 1.0
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.base.requires_grad_(False)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank * self.multiplier

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + update * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """A plain nn.Linear with the adapter folded into the weights (zero inference overhead)."""
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged = merged.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        merged.weight.copy_(self.base.weight + self.delta_weight().to(self.base.weight.dtype))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def _targets(model: nn.Module, names: Sequence[str], cls) -> List[tuple]:
    found = []
    for _, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, cls) and child_name in names:
                found.append((module, child_name, child))
    return found


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    targets: Sequence[str] = DEFAULT_TARGETS,
    dropout: float = 0.0,
) -> int:
    """Wrap every nn.Linear whose attribute name is in `targets`. Returns the number of adapters added."""
    found = _targets(model, targets, nn.Linear)
    for parent, name, linear in found:
        wrapped = LoRALinear(linear, rank, alpha, dropout).to(linear.weight.device)
        setattr(parent, name, wrapped)
    return len(found)


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def set_lora_scale(model: nn.Module, scale: float) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.multiplier = scale


def merge_lora(model: nn.Module) -> int:
    """Fold all adapters into their base weights in place and remove the wrappers."""
    found = _all_lora(model)
    for parent, name, lora in found:
        setattr(parent, name, lora.merged_linear())
    return len(found)


def unload_lora(model: nn.Module) -> int:
    """Remove adapters, restoring the untouched base layers."""
    found = _all_lora(model)
    for parent, name, lora in found:
        setattr(parent, name, lora.base)
    return len(found)


def _all_lora(model: nn.Module) -> List[tuple]:
    return [
        (module, child_name, child)
        for _, module in model.named_modules()
        for child_name, child in module.named_children()
        if isinstance(child, LoRALinear)
    ]


def save_lora(model: nn.Module, path: str, rank: int, alpha: float, targets: Iterable[str] = DEFAULT_TARGETS) -> None:
    save_checkpoint({"state": lora_state_dict(model), "rank": rank, "alpha": alpha, "targets": list(targets)}, path)


def load_lora(model: nn.Module, path: str, scale: float = 1.0, merge: bool = True) -> None:
    """Inject adapters described by a saved file into `model`, optionally merging them."""
    blob = load_checkpoint(path)
    inject_lora(model, blob["rank"], blob["alpha"], blob["targets"])
    missing, unexpected = model.load_state_dict(blob["state"], strict=False)
    if unexpected or any("lora_" in k for k in missing):
        raise RuntimeError(f"LoRA file does not match the model: missing={missing[:3]} unexpected={unexpected[:3]}")
    set_lora_scale(model, scale)
    if merge:
        merge_lora(model)
