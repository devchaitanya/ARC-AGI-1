"""Efficient transformer layers for ARC grids."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class RotaryEmbedding3D(nn.Module):
    """3D RoPE over row, column, and pair/demo index."""

    def __init__(self, head_dim: int, max_row: int = 30, max_col: int = 30, max_pair: int = 16, base: float = 10000.0):
        super().__init__()
        if head_dim < 6:
            raise ValueError("head_dim must be at least 6 for 3D RoPE")
        sizes = [head_dim // 3, head_dim // 3, head_dim - 2 * (head_dim // 3)]
        sizes = [s if s % 2 == 0 else s - 1 for s in sizes]
        remainder = head_dim - sum(sizes)
        sizes[-1] += remainder
        if any(s < 2 or s % 2 for s in sizes):
            raise ValueError(f"Cannot split head_dim={head_dim} into even 3D RoPE chunks")
        self.sizes = tuple(sizes)
        self.register_buffer("row_freqs", self._freqs(max_row, sizes[0], base), persistent=False)
        self.register_buffer("col_freqs", self._freqs(max_col, sizes[1], base), persistent=False)
        self.register_buffer("pair_freqs", self._freqs(max_pair, sizes[2], base), persistent=False)

    @staticmethod
    def _freqs(length: int, dim: int, base: float) -> torch.Tensor:
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(length).float(), inv)
        return torch.cat([freqs, freqs], dim=-1)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    def _apply_axis(self, x: torch.Tensor, freqs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        angles = freqs[pos].unsqueeze(1)
        return x * angles.cos() + self._rotate_half(x) * angles.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        row_pos: torch.Tensor,
        col_pos: torch.Tensor,
        pair_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunks_q = q.split(self.sizes, dim=-1)
        chunks_k = k.split(self.sizes, dim=-1)
        positions = (row_pos, col_pos, pair_pos)
        freqs = (self.row_freqs, self.col_freqs, self.pair_freqs)
        q_out = [self._apply_axis(c, f, p.clamp_max(f.shape[0] - 1)) for c, f, p in zip(chunks_q, freqs, positions)]
        k_out = [self._apply_axis(c, f, p.clamp_max(f.shape[0] - 1)) for c, f, p in zip(chunks_k, freqs, positions)]
        return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)


class MultiHeadAttention3D(nn.Module):
    """FlashAttention-backed self-attention with 3D RoPE."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0, max_grid: int = 30, max_pair: int = 16):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding3D(self.head_dim, max_grid, max_grid, max_pair)

    def forward(
        self,
        x: torch.Tensor,
        row_pos: torch.Tensor,
        col_pos: torch.Tensor,
        pair_pos: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.rope(q, k, row_pos, col_pos, pair_pos)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(out.transpose(1, 2).contiguous().view(bsz, seq_len, dim))


class TransformerBlock3D(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, max_grid: int = 30, max_pair: int = 16):
        super().__init__()
        hidden = int(dim * mlp_ratio * 2 / 3)
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention3D(dim, heads, dropout, max_grid, max_pair)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden, dropout)

    def forward(self, x: torch.Tensor, row_pos: torch.Tensor, col_pos: torch.Tensor, pair_pos: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), row_pos, col_pos, pair_pos, key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
