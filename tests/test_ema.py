import torch
import torch.nn as nn

from sd.ema import EMA


def test_ema_tracks_and_swaps_weights():
    model = nn.Linear(2, 2)
    ema = EMA(model, decay=0.9, warmup=False)
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    expected = 0.9 * original + 0.1 * model.weight.detach()
    assert torch.allclose(ema.shadow["weight"], expected, atol=1e-6)

    trained = model.weight.detach().clone()
    with ema.average_parameters(model):
        assert torch.allclose(model.weight, expected, atol=1e-6)
    assert torch.allclose(model.weight, trained)  # restored on exit


def test_ema_warmup_and_state_roundtrip():
    model = nn.Linear(2, 2)
    ema = EMA(model, decay=0.999)
    assert ema.current_decay() == 0.1  # (1 + 0) / (10 + 0)
    for _ in range(5):
        ema.update(model)
    other = EMA(nn.Linear(2, 2), decay=0.999)
    other.load_state_dict(ema.state_dict())
    assert other.num_updates == 5
    assert torch.equal(other.shadow["weight"], ema.shadow["weight"])


def test_ema_skips_frozen_parameters():
    model = nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    assert set(EMA(model).shadow) == {"weight"}
