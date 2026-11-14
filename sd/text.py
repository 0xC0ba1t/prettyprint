"""A small CLIP-style causal text transformer producing per-token conditioning for cross-attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TextConfig


class TextAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0, "width must be divisible by heads"
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(b, n, c))


class TextLayer(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = TextAttention(width, heads)
        self.ln2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        h = self.fc1(self.ln2(x))
        return x + self.fc2(h * torch.sigmoid(1.702 * h))  # QuickGELU, as in CLIP


class TextEncoder(nn.Module):
    """token ids (B, L) -> hidden states (B, L, width)."""

    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position_embedding = nn.Parameter(torch.empty(cfg.max_length, cfg.width))
        self.layers = nn.ModuleList([TextLayer(cfg.width, cfg.heads) for _ in range(cfg.layers)])
        self.final_ln = nn.LayerNorm(cfg.width)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(ids) + self.position_embedding[: ids.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.final_ln(x)
