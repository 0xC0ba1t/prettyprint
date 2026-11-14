"""Byte-level BPE tokenizer trained on the dataset's own captions (no external vocab needed)."""
import os
from typing import Iterable, List, Sequence, Union

import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


class TextTokenizer:
    def __init__(self, tokenizer: Tokenizer, max_length: int = 77):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id(PAD)
        self.bos_id = tokenizer.token_to_id(BOS)
        self.eos_id = tokenizer.token_to_id(EOS)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train(cls, captions: Iterable[str], vocab_size: int = 8192, max_length: int = 77) -> "TextTokenizer":
        tok = Tokenizer(models.BPE(unk_token=UNK))
        tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(captions, trainer)
        return cls(tok, max_length)

    @classmethod
    def load(cls, path: str, max_length: int = 77) -> "TextTokenizer":
        return cls(Tokenizer.from_file(path), max_length)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tokenizer.save(path)

    def encode(self, text: str) -> List[int]:
        """[BOS] tokens [EOS], truncated to `max_length`, then right-padded with PAD."""
        ids = self.tokenizer.encode(text).ids[: self.max_length - 2]
        ids = [self.bos_id] + ids + [self.eos_id]
        return ids + [self.pad_id] * (self.max_length - len(ids))

    def __call__(self, texts: Union[str, Sequence[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)

    def decode(self, ids: Sequence[int]) -> str:
        keep = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.tokenizer.decode(keep)
