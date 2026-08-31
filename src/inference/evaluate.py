"""Evaluation helpers for ARC tasks."""

from __future__ import annotations

import torch

from src.utils.metrics import exact_match, pixel_accuracy


@torch.no_grad()
def predict_batch(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    outputs = model(batch)
    return outputs["logits"].argmax(dim=1)


@torch.no_grad()
def evaluate_loader(model, loader, device: torch.device | str = "cpu") -> dict[str, float]:
    device = torch.device(device)
    model.to(device).eval()
    exact_scores, pixel_scores = [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        pred = predict_batch(model, batch)
        exact_scores.append(exact_match(pred, batch["test_output"], batch["test_output_mask"]))
        pixel_scores.append(pixel_accuracy(pred, batch["test_output"], batch["test_output_mask"]))
    return {
        "exact_match": float(sum(exact_scores) / max(1, len(exact_scores))),
        "pixel_accuracy": float(sum(pixel_scores) / max(1, len(pixel_scores))),
    }


@torch.no_grad()
def predicted_shapes(outputs: dict, max_grid: int) -> list[tuple[int, int]]:
    """Decode the auxiliary shape head into ``(height, width)`` pairs."""

    heights = outputs["height_logits"].argmax(dim=-1).add(1).clamp(1, max_grid)
    widths = outputs["width_logits"].argmax(dim=-1).add(1).clamp(1, max_grid)
    return [(int(h), int(w)) for h, w in zip(heights.cpu(), widths.cpu())]


@torch.no_grad()
def predict_grids(model, batch: dict[str, torch.Tensor], use_shape_head: bool = True) -> list[list[list[int]]]:
    """Predict cropped ARC grids for a batch.

    The output canvas is cropped with the shape head when ``use_shape_head`` is
    set, and otherwise with the ground-truth ``output_shape`` entry.
    """

    model.eval()
    outputs = model(batch)
    preds = outputs["logits"].argmax(dim=1).cpu()
    max_grid = preds.shape[-1]
    if use_shape_head:
        shapes = predicted_shapes(outputs, max_grid)
    else:
        shapes = [(int(h), int(w)) for h, w in batch["output_shape"].cpu()]
    return [pred[:h, :w].tolist() for pred, (h, w) in zip(preds, shapes)]
