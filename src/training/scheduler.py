"""Learning-rate schedules."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05) -> LambdaLR:
    """Linear warmup followed by cosine decay."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, factor)


def wsd_schedule(optimizer: Optimizer, warmup_steps: int, stable_steps: int, decay_steps: int, min_lr_ratio: float = 0.1) -> LambdaLR:
    """Warmup-stable-decay schedule commonly used for transformer training."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        if step < warmup_steps + stable_steps:
            return 1.0
        progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
        return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * min(1.0, progress))

    return LambdaLR(optimizer, factor)
