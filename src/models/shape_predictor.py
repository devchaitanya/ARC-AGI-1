"""Auxiliary grid-shape prediction head."""

from __future__ import annotations

import torch
from torch import nn


class GridShapePredictor(nn.Module):
    def __init__(self, dim: int, max_grid: int = 30):
        super().__init__()
        self.max_grid = max_grid
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.height = nn.Linear(dim, max_grid)
        self.width = nn.Linear(dim, max_grid)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(pooled)
        return self.height(h), self.width(h)
