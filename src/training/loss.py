"""Loss functions for recursive deep supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def deep_supervision_loss(
    logits_by_step: list[torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.35,
) -> torch.Tensor:
    """Exponentially weighted CE over recurrent prediction steps."""

    if not logits_by_step:
        raise ValueError("logits_by_step cannot be empty")
    weights = torch.tensor([gamma**i for i in range(len(logits_by_step))], device=targets.device)
    weights = weights / weights.sum()
    safe_targets = targets.masked_fill(~mask, -100)
    loss = targets.new_tensor(0.0, dtype=torch.float32)
    for weight, logits in zip(weights, logits_by_step):
        loss = loss + weight * F.cross_entropy(logits, safe_targets, ignore_index=-100)
    return loss


def shape_loss(height_logits: torch.Tensor, width_logits: torch.Tensor, output_shape: torch.Tensor) -> torch.Tensor:
    """Auxiliary CE for output grid height/width."""

    height = output_shape[:, 0].clamp_min(1) - 1
    width = output_shape[:, 1].clamp_min(1) - 1
    return F.cross_entropy(height_logits, height) + F.cross_entropy(width_logits, width)
