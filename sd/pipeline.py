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
import torch.nn.functional as F

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
        mask: Optional[torch.Tensor] = None,
        init_latents: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, t in enumerate(timesteps):
            model_out = self._predict(latents, t, cond, uncond, guidance_scale)
            latents = sampler.step(model_out, t, latents, generator)
            if mask is not None:
                # Keep the known region on the forward-noised original; only the masked region is generated.
                t_prev = sampler.prev_timestep(t)
                known = self.schedule.add_noise(init_latents, noise, t_prev) if t_prev >= 0 else init_latents
                latents = mask * latents + (1.0 - mask) * known
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

    # ------------------------------------------------------------------ image -> image
    @torch.no_grad()
    def _encode_image(self, image: Image.Image, n: int, generator) -> torch.Tensor:
        x = pil_to_tensor(fit_image(image, self.image_size)).unsqueeze(0).to(self.device)
        with self._autocast():
            z = self.vae.encode(x).mode().float() * self.vae.scale_factor
        return z.expand(n, -1, -1, -1).contiguous()

    @torch.no_grad()
    def img2img(
        self,
        prompt: Prompt,
        image: Image.Image,
        strength: float = 0.7,
        negative_prompt: Prompt = "",
        num_images: int = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        sampler: str = "dpmpp",
        seed: Optional[int] = None,
        callback=None,
    ) -> List[Image.Image]:
        """Noise the encoded input to `strength` of the schedule, then denoise it under the new prompt."""
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        prompts = self._as_list(prompt, num_images)
        cond, uncond = self.encode_prompts(prompts, negative_prompt)
        generator = self._generator(seed)
        solver = make_sampler(sampler, self.schedule)
        solver.set_timesteps(num_inference_steps)
        timesteps = solver.timesteps_for_strength(strength)
        init = self._encode_image(image, len(prompts), generator)
        if not timesteps:  # strength 0: nothing to denoise
            return self._decode(init)
        latents = self.schedule.add_noise(init, self._initial_noise(len(prompts), generator), timesteps[0])
        latents = self._denoise(latents, solver, timesteps, cond, uncond, guidance_scale, generator, callback)
        return self._decode(latents)

    # ------------------------------------------------------------------ inpainting
    def _latent_mask(self, mask: Image.Image) -> torch.Tensor:
        """White = repaint. Max-pooled to latent resolution so any touched latent cell is regenerated."""
        m = pil_to_tensor(fit_image(mask.convert("L").convert("RGB"), self.image_size, Image.NEAREST))[:1]
        m = ((m + 1.0) / 2.0 > 0.5).float().unsqueeze(0)
        return F.adaptive_max_pool2d(m, self.latent_size).to(self.device)

    @torch.no_grad()
    def inpaint(
        self,
        prompt: Prompt,
        image: Image.Image,
        mask: Image.Image,
        strength: float = 1.0,
        negative_prompt: Prompt = "",
        num_images: int = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        sampler: str = "dpmpp",
        seed: Optional[int] = None,
        composite: bool = True,
        callback=None,
    ) -> List[Image.Image]:
        """Regenerate the white region of `mask` while keeping the rest of `image`.

        At every step the unmasked latents are reset to the original image noised to the matching level,
        so the model only has to be consistent with the surroundings. With `composite=True` the untouched
        pixels are also copied back from the source, avoiding any VAE round-trip loss outside the mask.
        """
        prompts = self._as_list(prompt, num_images)
        cond, uncond = self.encode_prompts(prompts, negative_prompt)
        generator = self._generator(seed)
        solver = make_sampler(sampler, self.schedule)
        solver.set_timesteps(num_inference_steps)
        timesteps = solver.timesteps_for_strength(strength)
        init = self._encode_image(image, len(prompts), generator)
        latent_mask = self._latent_mask(mask)
        noise = self._initial_noise(len(prompts), generator)
        latents = self.schedule.add_noise(init, noise, timesteps[0]) if timesteps else init
        latents = self._denoise(
            latents, solver, timesteps, cond, uncond, guidance_scale, generator, callback,
            mask=latent_mask, init_latents=init, noise=noise,
        )
        results = self._decode(latents)
        if composite:
            source = fit_image(image, self.image_size)
            pixel_mask = fit_image(mask.convert("L").convert("RGB"), self.image_size, Image.NEAREST).convert("L")
            results = [Image.composite(r, source, pixel_mask) for r in results]
        return results
