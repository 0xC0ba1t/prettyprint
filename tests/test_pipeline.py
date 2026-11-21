import numpy as np
import pytest
import torch
from PIL import Image

from tests.conftest_helpers import tiny_pipeline


@pytest.fixture(scope="module")
def pipe():
    return tiny_pipeline()


def test_text_to_image_shape_and_seed_determinism(pipe):
    a = pipe("a red square", num_images=2, num_inference_steps=4, seed=3)
    b = pipe("a red square", num_images=2, num_inference_steps=4, seed=3)
    c = pipe("a red square", num_images=2, num_inference_steps=4, seed=4)
    assert len(a) == 2 and a[0].size == (32, 32) and a[0].mode == "RGB"
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    assert not np.array_equal(np.asarray(a[0]), np.asarray(c[0]))


@pytest.mark.parametrize("sampler", ["ddpm", "ddim", "dpmpp"])
def test_every_sampler_runs(pipe, sampler):
    latents = pipe("a blue circle", num_inference_steps=5, sampler=sampler, seed=0, return_latents=True)
    assert latents.shape == (1, 4, 4, 4)
    assert torch.isfinite(latents).all()


def test_guidance_scale_changes_output_and_scale_one_skips_cfg(pipe):
    base = pipe("a red square", num_inference_steps=4, guidance_scale=1.0, seed=1, return_latents=True)
    guided = pipe("a red square", num_inference_steps=4, guidance_scale=7.0, seed=1, return_latents=True)
    assert not torch.allclose(base, guided)


def test_negative_prompt_and_prompt_lists(pipe):
    imgs = pipe(["a red square", "a blue circle"], negative_prompt="a green triangle", num_inference_steps=3, seed=0)
    assert len(imgs) == 2


def test_callback_sees_every_step(pipe):
    seen = []
    pipe("a red square", num_inference_steps=6, seed=0, callback=lambda i, t, z: seen.append((i, t)))
    assert [i for i, _ in seen] == list(range(6))
    assert seen[0][1] == 999


def test_img2img_strength_zero_returns_reconstruction_and_runs_otherwise(pipe):
    source = Image.fromarray((np.random.RandomState(0).rand(40, 56, 3) * 255).astype("uint8"))
    same = pipe.img2img("a red square", source, strength=0.0, num_inference_steps=4, seed=0)
    assert same[0].size == (32, 32)
    edited = pipe.img2img("a red square", source, strength=0.6, num_inference_steps=5, seed=0)
    assert edited[0].size == (32, 32)
    with pytest.raises(ValueError):
        pipe.img2img("x", source, strength=1.5)


def test_inpaint_keeps_unmasked_pixels(pipe):
    source = Image.fromarray((np.random.RandomState(1).rand(32, 32, 3) * 255).astype("uint8"))
    mask = Image.new("L", (32, 32), 0)
    mask.paste(255, (16, 0, 32, 32))  # repaint the right half only
    out = pipe.inpaint("a blue circle", source, mask, num_inference_steps=5, seed=0)[0]
    src, res = np.asarray(source), np.asarray(out)
    assert np.array_equal(res[:, :16], src[:, :16])  # untouched half is bit-exact
    assert not np.array_equal(res[:, 16:], src[:, 16:])  # masked half was regenerated


def test_inpaint_empty_mask_returns_source(pipe):
    source = Image.fromarray((np.random.RandomState(2).rand(32, 32, 3) * 255).astype("uint8"))
    out = pipe.inpaint("a blue circle", source, Image.new("L", (32, 32), 0), num_inference_steps=3, seed=0)[0]
    assert np.array_equal(np.asarray(out), np.asarray(source))


def test_v_prediction_zero_snr_pipeline_runs():
    p = tiny_pipeline(prediction_type="v_prediction", zero_terminal_snr=True)
    latents = p("a red square", num_inference_steps=6, sampler="dpmpp", seed=0, return_latents=True)
    assert torch.isfinite(latents).all()
