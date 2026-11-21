import copy

import pytest
import torch

from sd.config import UNetConfig
from sd.lora import (
    LoRALinear,
    inject_lora,
    load_lora,
    lora_parameters,
    lora_state_dict,
    merge_lora,
    save_lora,
    set_lora_scale,
    unload_lora,
)
from sd.unet import UNet


def make_unet():
    cfg = UNetConfig(model_channels=32, ch_mult=(1, 2), num_res_blocks=1, attention_levels=(1,), head_dim=16,
                     context_dim=24, groups=8)
    unet = UNet(cfg)
    torch.nn.init.normal_(unet.conv_out.weight, std=0.05)
    for m in unet.modules():
        if m.__class__.__name__ == "SpatialTransformer":
            torch.nn.init.normal_(m.proj_out.weight, std=0.05)
    return unet.eval()


def inputs():
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 4, 16, 16, generator=g), torch.tensor([10, 400]), torch.randn(2, 5, 24, generator=g)


def randomise_adapters(unet):
    for p in lora_parameters(unet):
        torch.nn.init.normal_(p, std=0.05)


def test_zero_init_adapter_is_identity():
    unet = make_unet()
    x, t, c = inputs()
    base = unet(x, t, c)
    assert inject_lora(unet, rank=4, alpha=4) > 0
    assert torch.allclose(unet(x, t, c), base, atol=1e-6)


def test_only_adapter_parameters_are_trainable():
    unet = make_unet()
    inject_lora(unet, rank=4, alpha=8)
    unet.requires_grad_(False)
    for p in lora_parameters(unet):
        p.requires_grad_(True)
    names = {n for n, p in unet.named_parameters() if p.requires_grad}
    assert names and all("lora_" in n for n in names)
    total = sum(p.numel() for p in unet.parameters())
    assert sum(p.numel() for p in lora_parameters(unet)) < 0.2 * total


def test_merge_equals_adapted_forward():
    unet = make_unet()
    inject_lora(unet, rank=4, alpha=8)
    randomise_adapters(unet)
    x, t, c = inputs()
    adapted = unet(x, t, c)
    n = merge_lora(unet)
    assert n > 0 and not any(isinstance(m, LoRALinear) for m in unet.modules())
    assert torch.allclose(unet(x, t, c), adapted, atol=1e-5)


def test_unload_restores_base_model():
    unet = make_unet()
    x, t, c = inputs()
    base = unet(x, t, c)
    inject_lora(unet, rank=4, alpha=8)
    randomise_adapters(unet)
    assert not torch.allclose(unet(x, t, c), base, atol=1e-4)
    unload_lora(unet)
    assert torch.allclose(unet(x, t, c), base, atol=1e-6)


def test_scale_zero_disables_adapter():
    unet = make_unet()
    x, t, c = inputs()
    base = unet(x, t, c)
    inject_lora(unet, rank=4, alpha=8)
    randomise_adapters(unet)
    set_lora_scale(unet, 0.0)
    assert torch.allclose(unet(x, t, c), base, atol=1e-6)


def test_save_load_roundtrip(tmp_path):
    src = make_unet()
    base_weights = copy.deepcopy(src.state_dict())
    inject_lora(src, rank=4, alpha=8)
    randomise_adapters(src)
    x, t, c = inputs()
    expected = src(x, t, c)
    path = str(tmp_path / "adapter.pt")
    save_lora(src, path, rank=4, alpha=8)
    assert all("lora_" in k for k in lora_state_dict(src))

    dst = make_unet()
    dst.load_state_dict(base_weights)
    load_lora(dst, path, scale=1.0, merge=False)
    assert torch.allclose(dst(x, t, c), expected, atol=1e-6)

    merged = make_unet()
    merged.load_state_dict(base_weights)
    load_lora(merged, path, merge=True)
    assert torch.allclose(merged(x, t, c), expected, atol=1e-5)


def test_load_rejects_mismatched_model(tmp_path):
    src = make_unet()
    inject_lora(src, rank=4, alpha=8, targets=("to_q",))
    path = str(tmp_path / "q_only.pt")
    save_lora(src, path, rank=4, alpha=8, targets=("to_q", "to_k"))  # claims more targets than it holds
    with pytest.raises(RuntimeError):
        load_lora(make_unet(), path)
