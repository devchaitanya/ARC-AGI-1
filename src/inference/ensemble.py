"""Test-time augmentation and ARC two-attempt voting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import torch

from src.data.augmentations import D8_TRANSFORMS
from src.data.dataset import ARCDatasetConfig, task_to_batch
from src.inference.evaluate import predict_grids


def vote_top2(grids: list[list[list[int]]]) -> list[list[list[int]]]:
    """Return the top two unique grid predictions by majority vote."""

    counter = Counter(tuple(tuple(row) for row in grid) for grid in grids)
    winners = [k for k, _ in counter.most_common(2)]
    if len(winners) == 1:
        winners.append(winners[0])
    return [[list(row) for row in winner] for winner in winners]


@torch.no_grad()
def tta_predict_two_attempts(
    model,
    task: dict,
    make_batch_fn: Callable[[dict], dict[str, torch.Tensor]] | None = None,
    device: torch.device | str = "cpu",
    dataset_config: ARCDatasetConfig | None = None,
    use_shape_head: bool = True,
) -> list[list[list[int]]]:
    """Run D8 TTA and inverse-transform predictions into ARC's two attempts.

    ``make_batch_fn`` defaults to :func:`src.data.dataset.task_to_batch`, which
    pads and stacks a raw ARC task into a single-sample batch.
    """

    config = dataset_config or ARCDatasetConfig(augment=False)
    make_batch_fn = make_batch_fn or (lambda t: task_to_batch(t, config))
    model.to(device).eval()
    predictions: list[list[list[int]]] = []
    for transform in D8_TRANSFORMS:
        aug_task = {
            **{k: v for k, v in task.items() if k not in {"train", "test"}},
            "train": [{"input": transform.apply(p["input"]), "output": transform.apply(p["output"])} for p in task["train"]],
            "test": [{"input": transform.apply(task["test"][0]["input"])}],
        }
        batch = make_batch_fn(aug_task)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        pred = predict_grids(model, batch, use_shape_head=use_shape_head)[0]
        predictions.append(transform.inverse(pred))
    return vote_top2(predictions)
