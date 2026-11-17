"""Forward-noising schedule and reverse-process samplers (DDPM, DDIM, DPM-Solver++)."""
import math
from typing import List, Optional, Union

import numpy as np
import torch

from .config import DiffusionConfig

TimestepLike = Union[int, torch.Tensor]


def make_betas(schedule: str, num_timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    elif schedule == "scaled_linear":  # the Stable Diffusion schedule
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float64) ** 2
    elif schedule == "cosine":  # Nichol & Dhariwal 2021

        def alpha_bar(t: float) -> float:
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = torch.tensor(
            [min(1 - alpha_bar((i + 1) / num_timesteps) / alpha_bar(i / num_timesteps), 0.999) for i in range(num_timesteps)],
            dtype=torch.float64,
        )
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return betas


class NoiseSchedule:
    """Holds the cumulative alphas and implements the forward process q(x_t | x_0)."""

    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg
        self.num_timesteps = cfg.num_timesteps
        self.prediction_type = cfg.prediction_type
        if self.prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        betas = make_betas(cfg.beta_schedule, cfg.num_timesteps, cfg.beta_start, cfg.beta_end)
        self.betas = betas.float()
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)  # float64 for accurate scalar lookups

    def alpha_bar(self, t: int) -> float:
        """Cumulative alpha at integer timestep t; t < 0 means 'fully denoised' (alpha_bar = 1)."""
        return 1.0 if t < 0 else float(self.alphas_cumprod[int(t)])

    def _coeffs(self, t: TimestepLike, like: torch.Tensor):
        acp = self.alphas_cumprod.to(device=like.device, dtype=torch.float32)
        t = torch.as_tensor(t, device=like.device).long()
        if t.dim() == 0:
            t = t.expand(like.shape[0])
        a = acp[t].sqrt().view(-1, *([1] * (like.dim() - 1)))
        s = (1.0 - acp[t]).sqrt().view(-1, *([1] * (like.dim() - 1)))
        return a.to(like.dtype), s.to(like.dtype)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: TimestepLike) -> torch.Tensor:
        """x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps."""
        a, s = self._coeffs(t, x0)
        return a * x0 + s * noise

    def get_velocity(self, x0: torch.Tensor, noise: torch.Tensor, t: TimestepLike) -> torch.Tensor:
        """v = sqrt(abar_t) eps - sqrt(1 - abar_t) x_0 (Salimans & Ho 2022)."""
        a, s = self._coeffs(t, x0)
        return a * noise - s * x0

    def training_target(self, x0: torch.Tensor, noise: torch.Tensor, t: TimestepLike) -> torch.Tensor:
        return noise if self.prediction_type == "epsilon" else self.get_velocity(x0, noise, t)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        acp = self.alphas_cumprod.to(device=t.device, dtype=torch.float32)[t.long()]
        return acp / (1.0 - acp)

    def predict_x0_eps(self, model_output: torch.Tensor, x_t: torch.Tensor, t: int):
        """Convert a raw network output into (x0 estimate, eps estimate) for the active parameterisation."""
        a, s = math.sqrt(self.alpha_bar(t)), math.sqrt(1.0 - self.alpha_bar(t))
        if self.prediction_type == "epsilon":
            return (x_t - s * model_output) / a, model_output
        return a * x_t - s * model_output, s * x_t + a * model_output


def randn_like(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Gaussian noise shaped like x. A CPU generator keeps results identical across devices."""
    return torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)


class Sampler:
    """Base class: owns the discretised timestep list and the previous-timestep lookup."""

    def __init__(self, schedule: NoiseSchedule):
        self.schedule = schedule
        self.timesteps: List[int] = []
        self._index = {}

    def set_timesteps(self, num_inference_steps: int) -> List[int]:
        """'Trailing' spacing: always starts at t = T-1 and ends near 0 (required for zero-terminal-SNR)."""
        T = self.schedule.num_timesteps
        if not 1 <= num_inference_steps <= T:
            raise ValueError(f"num_inference_steps must be in [1, {T}]")
        ts = np.round(T * (1.0 - np.arange(num_inference_steps) / num_inference_steps)).astype(int) - 1
        self.timesteps = [int(t) for t in ts]
        self._index = {t: i for i, t in enumerate(self.timesteps)}
        self.reset()
        return self.timesteps

    def reset(self) -> None:
        """Clear any multistep history (called whenever a new run begins)."""

    def prev_timestep(self, t: int) -> int:
        """Next timestep in the schedule, or -1 after the last one."""
        i = self._index[int(t)]
        return self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1

    def timesteps_for_strength(self, strength: float) -> List[int]:
        """Tail of the schedule used by img2img: strength 1.0 = all steps, 0.0 = none."""
        n = len(self.timesteps)
        keep = min(int(n * strength), n)
        return self.timesteps[n - keep :]

    def step(self, model_output: torch.Tensor, t: int, sample: torch.Tensor, generator: Optional[torch.Generator] = None):
        raise NotImplementedError


class DDPMSampler(Sampler):
    """Ancestral sampling from the true posterior q(x_{t-1} | x_t, x_0)."""

    def step(self, model_output, t, sample, generator=None):
        t_prev = self.prev_timestep(t)
        acp_t, acp_prev = self.schedule.alpha_bar(t), self.schedule.alpha_bar(t_prev)
        alpha_t = acp_t / acp_prev  # effective alpha over the (possibly strided) step
        beta_t = 1.0 - alpha_t
        x0, _ = self.schedule.predict_x0_eps(model_output, sample, t)
        mean = (math.sqrt(acp_prev) * beta_t / (1 - acp_t)) * x0 + (math.sqrt(alpha_t) * (1 - acp_prev) / (1 - acp_t)) * sample
        if t_prev < 0:
            return mean
        variance = max((1 - acp_prev) / (1 - acp_t) * beta_t, 1e-20)
        return mean + math.sqrt(variance) * randn_like(sample, generator)


SAMPLERS = {"ddpm": DDPMSampler}


def make_sampler(name: str, schedule: NoiseSchedule, **kwargs) -> Sampler:
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler '{name}'. Available: {sorted(SAMPLERS)}")
    return SAMPLERS[name](schedule, **kwargs)
