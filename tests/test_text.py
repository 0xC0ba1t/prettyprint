import torch

from sd.config import TextConfig
from sd.text import TextEncoder
from sd.tokenizer import TextTokenizer

CAPTIONS = [
    "a small blue water monster",
    "a red fire dragon with wings",
    "a cute yellow creature with big eyes",
    "a green plant monster with a bulb",
] * 8


def make_tokenizer(max_length=16):
    return TextTokenizer.train(CAPTIONS, vocab_size=300, max_length=max_length)


def test_tokenizer_shapes_and_special_tokens():
    tok = make_tokenizer()
    ids = tok(["a blue monster", ""])
    assert ids.shape == (2, 16)
    assert ids[0, 0] == tok.bos_id
    assert tok.eos_id in ids[0].tolist()
    assert ids[1, 0] == tok.bos_id and ids[1, 1] == tok.eos_id  # empty prompt = unconditional
    assert ids[0, -1] == tok.pad_id


def test_tokenizer_truncates_long_text():
    tok = make_tokenizer(max_length=8)
    ids = tok("a " * 100)
    assert ids.shape == (1, 8)
    assert ids[0, 0] == tok.bos_id and ids[0, -1] == tok.eos_id


def test_tokenizer_roundtrip(tmp_path):
    tok = make_tokenizer()
    path = str(tmp_path / "tok.json")
    tok.save(path)
    again = TextTokenizer.load(path, max_length=16)
    assert torch.equal(tok(CAPTIONS[:3]), again(CAPTIONS[:3]))
    assert "blue" in again.decode(tok.encode("a blue monster"))


def test_text_encoder_output_shape():
    cfg = TextConfig(vocab_size=300, max_length=16, width=32, layers=2, heads=4)
    enc = TextEncoder(cfg)
    out = enc(torch.randint(0, 300, (3, 16)))
    assert out.shape == (3, 16, 32)


def test_text_encoder_is_causal():
    cfg = TextConfig(vocab_size=300, max_length=16, width=32, layers=2, heads=4)
    enc = TextEncoder(cfg).eval()
    a = torch.randint(0, 300, (1, 16))
    b = a.clone()
    b[0, 10:] = (b[0, 10:] + 1) % 300  # change only the future tokens
    out_a, out_b = enc(a), enc(b)
    assert torch.allclose(out_a[:, :10], out_b[:, :10], atol=1e-5)
    assert not torch.allclose(out_a[:, 10:], out_b[:, 10:])
