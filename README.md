# prettyprint

A from-scratch **latent diffusion (Stable Diffusion style) text-to-image model** in PyTorch, sized so that the
whole pipeline - training *and* inference - fits a free **Google Colab T4** (16 GB, fp16).

Everything is implemented here: the KL autoencoder, a BPE tokenizer, a CLIP-style text transformer, a
cross-attention UNet, the noise schedules and three samplers, classifier-free guidance, EMA, LoRA, img2img and
inpainting. No `diffusers` or `transformers` dependency.

## Architecture

```
image 128x128 --VAE enc--> latent 4x16x16 --(+noise, t)--> UNet(text cond) --> eps / v
caption --BPE--> tokens --text transformer--> (77 x 256) context  ---cross-attention--^
```

| Part | File | Notes |
|------|------|-------|
| KL autoencoder | `sd/vae.py`, `sd/losses.py` | 8x downsample, mid-block attention, L1 + VGG perceptual + KL |
| Tokenizer | `sd/tokenizer.py` | byte-level BPE trained on the dataset's captions |
| Text encoder | `sd/text.py` | 6-layer causal transformer, QuickGELU, 77 tokens |
| UNet | `sd/unet.py`, `sd/attention.py` | ResBlocks + spatial transformers (self, cross, GEGLU), skip connections, gradient checkpointing |
| Noise + samplers | `sd/schedulers.py` | linear / scaled-linear / cosine betas, epsilon and **v-prediction**, **zero-terminal-SNR**, DDPM, DDIM, **DPM-Solver++ (2M)** |
| Training | `sd/trainer.py` | fp16 + GradScaler, grad accumulation, cosine LR, **EMA**, **Min-SNR weighting**, caption dropout for CFG, atomic resumable checkpoints |
| Inference | `sd/pipeline.py` | txt2img, **img2img**, **inpainting**, classifier-free guidance |
| Adapters | `sd/lora.py` | LoRA on attention projections: inject, merge, unload, save/load |

### T4 design choices

* **Cached latents** - the dataset is pushed through the frozen VAE once (with its mirror image); the diffusion
  model trains on 4x16x16 tensors and never runs the VAE again.
* **fp16 autocast** (the T4 has no bf16) with loss computed in fp32; attention via `scaled_dot_product_attention`.
* **~42M-parameter UNet** (plus a 21M VAE and 7M text encoder), small enough for batch 32 without checkpointing; `gradient_checkpointing: true` trades speed for memory.
* **Resumable**: checkpoints are written atomically to Drive; re-running a command continues where it stopped.
* **DPM-Solver++** gives good samples in ~20-25 steps.

## Quick start (Colab)

Open `notebooks/colab.ipynb`, select a T4 runtime and run the cells. Or from a shell:

```bash
pip install -r requirements.txt
python scripts/train_vae.py     --config configs/t4_128.yaml
python scripts/prepare_data.py  --config configs/t4_128.yaml --vae runs/vae/vae.pt
python scripts/train_ldm.py     --config configs/t4_128.yaml --vae runs/vae/vae.pt
python scripts/sample.py --ldm runs/ldm/latest.pt --vae runs/vae/vae.pt --tokenizer cache/tokenizer.json \
    --prompt "a blue water monster" --num 4
```

The default dataset is [`diffusers/pokemon-gpt4-captions`](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions).
To use your own images set `data.folder` in the config: a directory of images with optional `name.txt` captions.

### img2img, inpainting, LoRA

```bash
python scripts/sample.py ... --prompt "a red dragon" --init in.png --strength 0.6        # img2img
python scripts/sample.py ... --prompt "a hat" --init in.png --mask mask.png               # inpaint (white = repaint)
python scripts/train_lora.py --ldm runs/ldm/latest.pt --vae runs/vae/vae.pt \
    --tokenizer cache/tokenizer.json --folder my_images/                                   # train adapters
python scripts/sample.py ... --prompt "..." --lora runs/lora/lora.pt --lora-scale 0.8     # use them
```

## Tests

```bash
pytest -q
```

CPU-only unit tests cover module shapes, VAE/KL maths, schedule identities, solver consistency (every deterministic
sampler recovers x0 with an oracle denoiser), EMA, LoRA merge equivalence, pipeline behaviour (seeding, inpaint keeps
unmasked pixels), an overfit-a-batch training test and checkpoint resume.

## Credits

Built following the ideas of the original Stable Diffusion / latent diffusion papers, plus
[this walkthrough](https://youtu.be/ZBKpAp_6TGI?si=QSHEOyptzAEMTUk8) and
[these slides](https://github.com/hkproj/pytorch-stable-diffusion/blob/main/Stable_Diffusion_Diagrams_V2.pdf).
