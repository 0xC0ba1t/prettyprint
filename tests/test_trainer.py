import os

import pytest
import torch
from torch.utils.data import Dataset

from sd.attention import SpatialTransformer
from sd.data import LatentCaptionDataset, cache_latents
from sd.lora import inject_lora
from sd.schedulers import NoiseSchedule
from sd.text import TextEncoder
from sd.tokenizer import TextTokenizer
from sd.trainer import LDMTrainer
from sd.unet import UNet
from sd.vae import AutoencoderKL
from tests.conftest_helpers import CAPTIONS, tiny_config


class FixedLatents(Dataset):
    """A handful of fixed latents: the model should be able to memorise them (overfit test)."""

    def __init__(self, n=8, size=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.z = torch.randn(n, 4, size, size, generator=g)
        self.captions = CAPTIONS[:n]

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        return {"latent": self.z[i], "caption": self.captions[i]}


def build(tmp_path, steps=60, **train):
    torch.manual_seed(0)
    cfg = tiny_config()
    cfg.train.batch_size = 8
    cfg.train.max_steps = steps
    cfg.train.warmup_steps = 5
    cfg.train.lr = 2e-3
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.train.log_every = 10
    cfg.train.ckpt_every = 20
    cfg.train.sample_every = 10_000
    for k, v in train.items():
        setattr(cfg.train, k, v)
    tok = TextTokenizer.train(CAPTIONS, vocab_size=300, max_length=cfg.text.max_length)
    return cfg, LDMTrainer(
        cfg, UNet(cfg.unet), TextEncoder(cfg.text), tok, NoiseSchedule(cfg.diffusion),
        FixedLatents(), torch.device("cpu"),
    )


def test_loss_decreases_when_overfitting_fixed_latents(tmp_path):
    _, trainer = build(tmp_path, steps=120, min_snr_gamma=0.0, cond_drop_prob=0.0)
    trainer.fit()
    first, last = sum(trainer.history[:10]) / 10, sum(trainer.history[-10:]) / 10
    assert last < 0.9 * first, (first, last)


def test_training_updates_weights_and_ema(tmp_path):
    _, trainer = build(tmp_path, steps=3)
    before = {k: v.clone() for k, v in trainer.unet.state_dict().items()}
    trainer.fit()
    assert any(not torch.equal(v, trainer.unet.state_dict()[k]) for k, v in before.items())
    assert trainer.ema.num_updates == 3


@pytest.mark.parametrize("prediction_type,zero_snr", [("epsilon", False), ("v_prediction", True)])
def test_min_snr_weighted_loss_is_finite(tmp_path, prediction_type, zero_snr):
    cfg = tiny_config(prediction_type=prediction_type, zero_terminal_snr=zero_snr)
    _, trainer = build(tmp_path, steps=2)
    trainer.schedule = NoiseSchedule(cfg.diffusion)
    loss = trainer.loss_on_batch({"latent": torch.randn(8, 4, 4, 4), "caption": CAPTIONS[:8]})
    assert torch.isfinite(loss)


def test_checkpoint_resume_continues_from_saved_step(tmp_path):
    _, trainer = build(tmp_path, steps=20)
    trainer.fit()
    assert os.path.exists(os.path.join(trainer.tc.out_dir, "latest.pt"))

    cfg, fresh = build(tmp_path, steps=30)
    assert fresh.maybe_resume() is True
    assert fresh.step == 20
    for (n, a), (_, b) in zip(trainer.unet.state_dict().items(), fresh.unet.state_dict().items()):
        assert torch.equal(a, b), n
    fresh.fit()
    assert fresh.step == 30


def test_lora_only_training_freezes_base_and_saves_small_checkpoint(tmp_path):
    cfg = tiny_config()
    cfg.train.out_dir = str(tmp_path / "lora")
    cfg.train.max_steps, cfg.train.batch_size, cfg.train.ckpt_every = 4, 8, 4
    cfg.train.ema_decay = 0.0
    unet = UNet(cfg.unet)
    torch.nn.init.normal_(unet.conv_out.weight, std=0.05)  # a pretrained base is never zero-initialised
    for m in unet.modules():
        if isinstance(m, SpatialTransformer):
            torch.nn.init.normal_(m.proj_out.weight, std=0.05)
    inject_lora(unet, rank=4, alpha=4)
    tok = TextTokenizer.train(CAPTIONS, vocab_size=300, max_length=cfg.text.max_length)
    base_before = {k: v.clone() for k, v in unet.state_dict().items() if "lora_" not in k}
    trainer = LDMTrainer(
        cfg, unet, TextEncoder(cfg.text), tok, NoiseSchedule(cfg.diffusion), FixedLatents(), torch.device("cpu"),
        lora_only=True,
    )
    trainer.fit()
    for k, v in unet.state_dict().items():
        if "lora_" not in k:
            assert torch.equal(v, base_before[k]), k
    assert any(p.abs().sum() > 0 for n, p in unet.named_parameters() if "lora_B" in n)  # adapters moved
    saved = torch.load(os.path.join(cfg.train.out_dir, "latest.pt"), weights_only=False)["model"]
    assert saved and all("lora_" in k for k in saved)


def test_cache_latents_roundtrip(tmp_path):
    from PIL import Image
    import numpy as np

    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(5):
        arr = (np.random.RandomState(i).rand(48, 40, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(folder / f"thing_{i}.png")
        (folder / f"thing_{i}.txt").write_text(f"caption number {i}")
    from sd.data import FolderRecords

    cfg = tiny_config()
    vae = AutoencoderKL(cfg.vae)
    path = str(tmp_path / "latents.pt")
    scale = cache_latents(vae, FolderRecords(str(folder)), 32, path, batch_size=2, device=torch.device("cpu"))
    assert scale > 0 and float(vae.scale_factor) == pytest.approx(scale)
    ds = LatentCaptionDataset(path)
    assert len(ds) == 5 and ds.captions[3] == "caption number 3"
    item = ds[0]
    assert item["latent"].shape == (4, 4, 4)
    assert abs(float(torch.cat([ds[i]["latent"].flatten() for i in range(5)]).std()) - 1.0) < 0.35
