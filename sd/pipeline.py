"""Inference pipeline: text-to-image with classifier-free guidance (plus img2img and inpainting)."""
from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

from .config import Config
from .schedulers import NoiseSchedule, Sampler, make_sampler
from .text import TextEncoder
from .tokenizer import TextTokenizer
from .unet import UNet
from .utils import fit_image, load_checkpoint, pil_to_tensor, tensor_to_pil
from .vae import AutoencoderKL

Prompt = Union[str, Sequence[str]]


class LatentDiffusionPipeline:
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: nn.Module,
        tokenizer: TextTokenizer,
        unet: nn.Module,
        schedule: NoiseSchedule,
        image_size: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = vae.to(self.device).eval()
        self.text_encoder = text_encoder.to(self.device).eval()
        self.unet = unet.to(self.device).eval()
        self.tokenizer, self.schedule = tokenizer, schedule
        self.image_size = image_size
        self.latent_size = image_size // vae.downsample_factor
        self.latent_channels = vae.cfg.latent_channels

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoints(
        cls,
        ldm_ckpt: str,
        vae_ckpt: str,
        tokenizer_path: str,
        device: Optional[torch.device] = None,
        use_ema: bool = True,
    ) -> "LatentDiffusionPipeline":
        ckpt = load_checkpoint(ldm_ckpt)
        cfg = Config.from_dict(ckpt["config"])
        unet, text = UNet(cfg.unet), TextEncoder(cfg.text)
        modules = nn.ModuleDict({"unet": unet, "text": text})
        modules.load_state_dict(ckpt["model"])
        if use_ema and ckpt.get("ema"):
            params = dict(modules.named_parameters())
            with torch.no_grad():
                for name, value in ckpt["ema"]["shadow"].items():
                    params[name].copy_(value)
        vae_blob = load_checkpoint(vae_ckpt)
        vae = AutoencoderKL(Config.from_dict(vae_blob["config"]).vae)
        vae.load_state_dict(vae_blob["model"])
        vae.scale_factor.fill_(ckpt["scale_factor"])
        tokenizer = TextTokenizer.load(tokenizer_path, cfg.text.max_length)
        schedule = NoiseSchedule(cfg.diffusion)
        return cls(vae, text, tokenizer, unet, schedule, cfg.data.image_size, device)

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    @staticmethod
    def _generator(seed: Optional[int]) -> Optional[torch.Generator]:
        return None if seed is None else torch.Generator().manual_seed(seed)  # CPU: reproducible anywhere

    @staticmethod
    def _as_list(prompt: Prompt, n: int) -> List[str]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        return prompts * n if len(prompts) == 1 else prompts

    @torch.no_grad()
    def encode_prompts(self, prompts: List[str], negative: Prompt = "") -> Tuple[torch.Tensor, torch.Tensor]:
        negatives = self._as_list(negative, len(prompts))
        ids = self.tokenizer(prompts + negatives).to(self.device)
        with self._autocast():
            emb = self.text_encoder(ids).float()
        return emb[: len(prompts)], emb[len(prompts) :]

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> List[Image.Image]:
        with self._autocast():
            images = self.vae.decode_latent(latents)
        return [tensor_to_pil(img) for img in images.float()]

    @torch.no_grad()
    def _predict(self, latents, t: int, cond, uncond, guidance_scale: float) -> torch.Tensor:
        """Classifier-free guidance: eps_u + w (eps_c - eps_u), computed in one batched forward pass."""
        n = latents.shape[0]
        use_cfg = guidance_scale != 1.0
        x = torch.cat([latents, latents]) if use_cfg else latents
        ctx = torch.cat([cond, uncond]) if use_cfg else cond
        tt = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        with self._autocast():
            out = self.unet(x, tt, ctx).float()
        if not use_cfg:
            return out
        out_c, out_u = out[:n], out[n:]
        return out_u + guidance_scale * (out_c - out_u)

    @torch.no_grad()
    def _denoise(
        self,
        latents: torch.Tensor,
        sampler: Sampler,
        timesteps: List[int],
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        for i, t in enumerate(timesteps):
            model_out = self._predict(latents, t, cond, uncond, guidance_scale)
            latents = sampler.step(model_out, t, latents, generator)
            if callback is not None:
                callback(i, t, latents)
        return latents

    def _initial_noise(self, n: int, generator) -> torch.Tensor:
        shape = (n, self.latent_channels, self.latent_size, self.latent_size)
        return torch.randn(shape, generator=generator).to(self.device)

    # ------------------------------------------------------------------ text -> image
    @torch.no_grad()
    def __call__(
        self,
        prompt: Prompt,
        negative_prompt: Prompt = "",
        num_images: int = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        sampler: str = "dpmpp",
        seed: Optional[int] = None,
        return_latents: bool = False,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ):
        prompts = self._as_list(prompt, num_images)
        cond, uncond = self.encode_prompts(prompts, negative_prompt)
        generator = self._generator(seed)
        solver = make_sampler(sampler, self.schedule)
        timesteps = solver.set_timesteps(num_inference_steps)
        latents = self._initial_noise(len(prompts), generator)
        latents = self._denoise(latents, solver, timesteps, cond, uncond, guidance_scale, generator, callback)
        return latents if return_latents else self._decode(latents)
