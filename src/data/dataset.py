"""ARC task loading, padding, and batching."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .augmentations import D8_TRANSFORMS, sample_color_permutations, transform_task

PAD_VALUE = -1
NUM_COLORS = 10


@dataclass(frozen=True)
class ARCDatasetConfig:
    max_grid: int = 30
    max_demos: int = 4
    augment: bool = False
    color_permutations: int = 1
    seed: int = 42


def load_arc_tasks(path: str | Path) -> list[dict]:
    """Load ARC JSON files from one file or a directory."""

    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.json"))
    tasks = []
    for idx, file in enumerate(files):
        with file.open("r", encoding="utf-8") as fh:
            task = json.load(fh)
        task.setdefault("id", file.stem)
        task.setdefault("_task_id", idx)
        tasks.append(task)
    return tasks


def pad_grid(grid: Sequence[Sequence[int]], max_grid: int = 30, pad_value: int = PAD_VALUE) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Pad a rectangular ARC grid to ``max_grid x max_grid`` with a validity mask."""

    tensor = torch.as_tensor(grid, dtype=torch.long)
    h, w = tensor.shape
    if h > max_grid or w > max_grid:
        raise ValueError(f"Grid shape {(h, w)} exceeds max_grid={max_grid}")
    padded = torch.full((max_grid, max_grid), pad_value, dtype=torch.long)
    mask = torch.zeros((max_grid, max_grid), dtype=torch.bool)
    padded[:h, :w] = tensor
    mask[:h, :w] = True
    return padded, mask, (h, w)


class ARCTaskDataset(Dataset):
    """TRM-style ARC task dataset with offline D8 and sampled color variants."""

    def __init__(self, tasks_or_path: str | Path | list[dict], config: ARCDatasetConfig | None = None):
        self.config = config or ARCDatasetConfig()
        base_tasks = load_arc_tasks(tasks_or_path) if not isinstance(tasks_or_path, list) else tasks_or_path
        rng = random.Random(self.config.seed)
        self.samples: list[dict] = []
        for task_id, task in enumerate(base_tasks):
            task = {**task, "_task_id": task.get("_task_id", task_id)}
            transforms = D8_TRANSFORMS if self.config.augment else D8_TRANSFORMS[:1]
            perms = sample_color_permutations(self.config.color_permutations, rng) if self.config.augment else [None]
            for transform in transforms:
                for perm in perms:
                    self.samples.append(transform_task(task, transform, perm))
        self.num_base_tasks = len(base_tasks)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        task = self.samples[idx]
        max_grid = self.config.max_grid
        max_demos = self.config.max_demos

        demo_inputs, demo_outputs, demo_input_masks, demo_output_masks = [], [], [], []
        for pair in task["train"][:max_demos]:
            inp, inp_mask, _ = pad_grid(pair["input"], max_grid=max_grid)
            out, out_mask, _ = pad_grid(pair["output"], max_grid=max_grid)
            demo_inputs.append(inp)
            demo_outputs.append(out)
            demo_input_masks.append(inp_mask)
            demo_output_masks.append(out_mask)

        n_demos = len(demo_inputs)
        empty_grid = torch.full((max_grid, max_grid), PAD_VALUE, dtype=torch.long)
        empty_mask = torch.zeros((max_grid, max_grid), dtype=torch.bool)
        while len(demo_inputs) < max_demos:
            demo_inputs.append(empty_grid.clone())
            demo_outputs.append(empty_grid.clone())
            demo_input_masks.append(empty_mask.clone())
            demo_output_masks.append(empty_mask.clone())

        test_pair = task["test"][0]
        test_input, test_input_mask, input_shape = pad_grid(test_pair["input"], max_grid=max_grid)
        if "output" in test_pair:
            test_output, test_output_mask, output_shape = pad_grid(test_pair["output"], max_grid=max_grid)
        else:
            test_output = empty_grid.clone()
            test_output_mask = empty_mask.clone()
            output_shape = input_shape

        return {
            "demo_inputs": torch.stack(demo_inputs),
            "demo_outputs": torch.stack(demo_outputs),
            "demo_input_masks": torch.stack(demo_input_masks),
            "demo_output_masks": torch.stack(demo_output_masks),
            "test_input": test_input,
            "test_output": test_output,
            "test_input_mask": test_input_mask,
            "test_output_mask": test_output_mask,
            "n_demos": torch.tensor(n_demos, dtype=torch.long),
            "task_id": torch.tensor(task.get("_task_id", 0), dtype=torch.long),
            "output_shape": torch.tensor(output_shape, dtype=torch.long),
        }


def task_to_batch(task: dict, config: ARCDatasetConfig | None = None) -> dict[str, torch.Tensor]:
    """Build a single-sample batch (leading dim 1) from a raw ARC task dict."""

    config = config or ARCDatasetConfig(augment=False)
    dataset = ARCTaskDataset([task], config)
    return {k: v.unsqueeze(0) for k, v in dataset[0].items()}


def collate_tasks(tasks: Sequence[dict], config: ARCDatasetConfig | None = None) -> dict[str, torch.Tensor]:
    """Stack several raw ARC tasks into one batch."""

    if not tasks:
        raise ValueError("tasks cannot be empty")
    config = config or ARCDatasetConfig(augment=False)
    dataset = ARCTaskDataset(list(tasks), config)
    samples = [dataset[i] for i in range(len(dataset))]
    return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}
