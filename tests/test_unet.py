import torch

from sd.config import UNetConfig
from sd.embeddings import timestep_embedding
from sd.unet import UNet
from sd.utils import count_parameters


def tiny_cfg(**kw):
    base = dict(model_channels=32, ch_mult=(1, 2), num_res_blocks=1, attention_levels=(1,), head_dim=16,
                context_dim=24, groups=8)
    base.update(kw)
    return UNetConfig(**base)


def test_timestep_embedding_shape_and_distinct():
    emb = timestep_embedding(torch.tensor([0, 1, 500]), 32)
    assert emb.shape == (3, 32)
    assert not torch.allclose(emb[0], emb[1])
    assert timestep_embedding(torch.tensor([3]), 33).shape == (1, 33)  # odd dims are padded


def test_unet_output_shape_and_zero_init():
    unet = UNet(tiny_cfg())
    x, t, ctx = torch.randn(2, 4, 16, 16), torch.tensor([3, 700]), torch.randn(2, 7, 24)
    out = unet(x, t, ctx)
    assert out.shape == x.shape
    assert out.abs().max() == 0  # conv_out is zero-initialised


def unzero(unet, std=0.05):
    """Break the zero-initialisation so outputs and gradients are informative."""
    torch.nn.init.normal_(unet.conv_out.weight, std=std)
    for m in unet.modules():
        if m.__class__.__name__ == "SpatialTransformer":
            torch.nn.init.normal_(m.proj_out.weight, std=std)


def test_unet_depends_on_context_and_timestep():
    torch.manual_seed(0)
    unet = UNet(tiny_cfg())
    unzero(unet)
    x = torch.randn(1, 4, 16, 16)
    a = unet(x, torch.tensor([10]), torch.randn(1, 7, 24))
    b = unet(x, torch.tensor([10]), torch.randn(1, 7, 24))
    c = unet(x, torch.tensor([900]), torch.randn(1, 7, 24))
    assert not torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_gradient_checkpointing_matches_plain_backward():
    torch.manual_seed(0)
    plain = UNet(tiny_cfg())
    unzero(plain)
    ckpt = UNet(tiny_cfg(gradient_checkpointing=True))
    ckpt.load_state_dict(plain.state_dict())
    x, t, ctx = torch.randn(2, 4, 16, 16), torch.tensor([5, 50]), torch.randn(2, 7, 24)
    plain(x, t, ctx).square().mean().backward()
    ckpt(x, t, ctx).square().mean().backward()
    for (n, p), (_, q) in zip(plain.named_parameters(), ckpt.named_parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-5), n


def test_default_unet_is_t4_sized():
    n = count_parameters(UNet(UNetConfig()))
    assert 30e6 < n < 150e6, n
