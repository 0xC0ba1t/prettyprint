import pytest
import torch

from sd.blocks import Downsample, ResnetBlock, Upsample
from sd.config import VAEConfig
from sd.losses import VAELoss
from sd.vae import AutoencoderKL, DiagonalGaussian


def tiny_cfg():
    return VAEConfig(base_channels=16, ch_mult=(1, 2, 2, 2), num_res_blocks=1, groups=8)


def test_resnet_block_shapes_and_temb():
    block = ResnetBlock(16, 32, temb_channels=64, groups=8)
    x, temb = torch.randn(2, 16, 8, 8), torch.randn(2, 64)
    assert block(x, temb).shape == (2, 32, 8, 8)
    assert block(x).shape == (2, 32, 8, 8)  # temb is optional


def test_down_up_sample_shapes():
    x = torch.randn(1, 8, 16, 16)
    assert Downsample(8)(x).shape == (1, 8, 8, 8)
    assert Upsample(8)(x).shape == (1, 8, 32, 32)


def test_vae_roundtrip_shape_and_downsample_factor():
    vae = AutoencoderKL(tiny_cfg())
    x = torch.randn(2, 3, 64, 64)
    recon, posterior = vae(x)
    assert recon.shape == x.shape
    assert posterior.mean.shape == (2, 4, 8, 8)
    assert vae.downsample_factor == 8


def test_latent_scaling_is_invertible():
    vae = AutoencoderKL(tiny_cfg()).eval()
    x = torch.randn(1, 3, 32, 32)
    z = vae.encode_to_latent(x, sample=False)
    direct = vae.encode(x).mode() * vae.scale_factor
    assert torch.allclose(z, direct, atol=1e-6)
    assert vae.decode_latent(z).shape == x.shape


def test_kl_is_zero_for_standard_normal_posterior():
    moments = torch.zeros(2, 8, 4, 4)  # mean 0, logvar 0 -> N(0, I)
    assert torch.allclose(DiagonalGaussian(moments).kl(), torch.zeros(2))


def test_vae_loss_backward():
    cfg = tiny_cfg()
    vae = AutoencoderKL(cfg)
    x = torch.randn(2, 3, 32, 32)
    recon, posterior = vae(x)
    loss, logs = VAELoss(cfg)(x, recon, posterior)
    loss.backward()
    assert {"l1", "kl", "total"} <= set(logs)
    assert all(p.grad is not None for p in vae.parameters() if p.requires_grad)


def test_perceptual_loss_zero_for_identical_inputs():
    pytest.importorskip("torchvision")
    from sd.losses import VGGPerceptualLoss

    perc = VGGPerceptualLoss(pretrained=False)
    x = torch.randn(1, 3, 32, 32).clamp(-1, 1)
    assert perc(x, x).item() == pytest.approx(0.0, abs=1e-6)
