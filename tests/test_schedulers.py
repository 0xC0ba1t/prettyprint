import pytest
import torch

from sd.config import DiffusionConfig
from sd.schedulers import NoiseSchedule, make_betas, make_sampler, rescale_zero_terminal_snr


def schedule(**kw):
    return NoiseSchedule(DiffusionConfig(**kw))


@pytest.mark.parametrize("name", ["linear", "scaled_linear", "cosine"])
def test_betas_are_valid_and_alpha_bar_decreases(name):
    betas = make_betas(name, 1000, 0.00085, 0.012)
    assert betas.shape == (1000,)
    assert (betas > 0).all() and (betas < 1).all()
    acp = torch.cumprod(1 - betas, 0)
    assert (acp[1:] < acp[:-1]).all()


def test_add_noise_matches_closed_form_and_batches_timesteps():
    s = schedule()
    x0, noise = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    t = torch.tensor([0, 500, 999])
    xt = s.add_noise(x0, noise, t)
    for i, ti in enumerate(t.tolist()):
        a = s.alpha_bar(ti)
        expected = a**0.5 * x0[i] + (1 - a) ** 0.5 * noise[i]
        assert torch.allclose(xt[i], expected, atol=1e-5)
    assert torch.allclose(s.add_noise(x0, noise, 500)[1], xt[1], atol=1e-6)  # scalar t broadcasts


def test_velocity_target_recovers_x0_and_eps():
    s = schedule(prediction_type="v_prediction")
    x0, noise = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    t = 321
    xt = s.add_noise(x0, noise, t)
    v = s.training_target(x0, noise, torch.tensor([t, t]))
    x0_hat, eps_hat = s.predict_x0_eps(v, xt, t)
    assert torch.allclose(x0_hat, x0, atol=1e-4)
    assert torch.allclose(eps_hat, noise, atol=1e-4)


def test_zero_terminal_snr():
    s = schedule(prediction_type="v_prediction", zero_terminal_snr=True)
    assert s.alpha_bar(999) == pytest.approx(0.0, abs=1e-12)
    assert s.alpha_bar(0) == pytest.approx(schedule().alpha_bar(0), rel=1e-4)
    with pytest.raises(ValueError):
        schedule(prediction_type="epsilon", zero_terminal_snr=True)


def test_timesteps_trailing_spacing_and_strength():
    sampler = make_sampler("ddim", schedule())
    ts = sampler.set_timesteps(50)
    assert len(ts) == 50 and ts[0] == 999 and ts[-1] == 19
    assert all(a > b for a, b in zip(ts, ts[1:]))
    assert sampler.prev_timestep(ts[-1]) == -1
    assert sampler.timesteps_for_strength(1.0) == ts
    assert sampler.timesteps_for_strength(0.5) == ts[25:]
    assert sampler.timesteps_for_strength(0.0) == []
    assert len(sampler.set_timesteps(7)) == 7


def oracle_run(name, prediction_type, steps, zero_snr=False, **kw):
    """Run a sampler with a perfect denoiser; a consistent solver must recover x0 exactly."""
    s = schedule(prediction_type=prediction_type, zero_terminal_snr=zero_snr)
    sampler = make_sampler(name, s, **kw)
    torch.manual_seed(0)
    x0, noise = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    ts = sampler.set_timesteps(steps)
    x = s.add_noise(x0, noise, ts[0])
    for t in ts:
        # a perfect network knows x0 and the *current* implied noise
        eps = (x - s.alpha_bar(t) ** 0.5 * x0) / (1 - s.alpha_bar(t)) ** 0.5
        out = eps if prediction_type == "epsilon" else s.get_velocity(x0, eps, t)
        x = sampler.step(out, t, x)
    return x, x0


@pytest.mark.parametrize("name", ["ddim", "dpmpp"])
@pytest.mark.parametrize("steps", [5, 20, 50])
def test_deterministic_samplers_recover_x0_with_oracle(name, steps):
    x, x0 = oracle_run(name, "epsilon", steps)
    assert torch.allclose(x, x0, atol=1e-3)


def test_samplers_recover_x0_with_v_prediction_and_zero_snr():
    for name in ("ddim", "dpmpp"):
        x, x0 = oracle_run(name, "v_prediction", 25, zero_snr=True)
        assert torch.allclose(x, x0, atol=1e-3), name


def test_ddpm_is_stochastic_but_seedable_and_finite():
    s = schedule()
    sampler = make_sampler("ddpm", s)
    ts = sampler.set_timesteps(20)
    x = torch.randn(1, 4, 8, 8)

    def run(seed):
        g = torch.Generator().manual_seed(seed)
        cur = x.clone()
        for t in ts:
            cur = sampler.step(torch.zeros_like(cur), t, cur, g)
        return cur

    a, b, c = run(1), run(1), run(2)
    assert torch.isfinite(a).all()
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_unknown_sampler_raises():
    with pytest.raises(ValueError):
        make_sampler("euler_nope", schedule())
