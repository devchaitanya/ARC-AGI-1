"""Optional test-time fine-tuning utilities."""

from __future__ import annotations

import copy

import torch
from torch.optim import AdamW

from src.training.loss import deep_supervision_loss, shape_loss


def adapt_on_support(
    model: torch.nn.Module,
    support_loader,
    steps: int = 20,
    lr: float = 1e-5,
    shape_loss_weight: float = 0.02,
    device: str | torch.device = "cuda",
) -> torch.nn.Module:
    """Return a task-adapted copy of ``model`` fine-tuned on support examples."""

    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    adapted = copy.deepcopy(model).to(device)
    adapted.train()
    optimizer = AdamW(adapted.parameters(), lr=lr, weight_decay=0.0)
    iterator = iter(support_loader)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(support_loader)
            batch = next(iterator)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = adapted(batch)
        loss = deep_supervision_loss(outputs["logits_by_step"], batch["test_output"], batch["test_output_mask"])
        loss = loss + shape_loss_weight * shape_loss(outputs["height_logits"], outputs["width_logits"], batch["output_shape"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
        optimizer.step()
    adapted.eval()
    return adapted
