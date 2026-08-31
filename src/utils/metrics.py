"""ARC metrics."""

from __future__ import annotations

import torch


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    if valid.sum().item() == 0:
        return 0.0
    return (pred[valid] == target[valid]).float().mean().item()


def exact_match(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool().flatten(1)
    eq = (pred == target).flatten(1)
    return (eq.logical_or(~valid).all(dim=1)).float().mean().item()


def crop_to_shape(grid: torch.Tensor, shape: torch.Tensor | tuple[int, int]) -> torch.Tensor:
    h, w = int(shape[0]), int(shape[1])
    return grid[..., :h, :w]
