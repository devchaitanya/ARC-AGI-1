"""ARC grid augmentation utilities."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DihedralTransform:
    """One of the eight D8 transforms represented as rotation plus mirror."""

    rotation: int = 0
    flip: bool = False

    def apply(self, grid: Sequence[Sequence[int]]) -> list[list[int]]:
        arr = np.asarray(grid, dtype=np.int64)
        arr = np.rot90(arr, self.rotation)
        if self.flip:
            arr = np.fliplr(arr)
        return arr.copy().tolist()

    def inverse(self, grid: Sequence[Sequence[int]]) -> list[list[int]]:
        arr = np.asarray(grid, dtype=np.int64)
        if self.flip:
            arr = np.fliplr(arr)
        arr = np.rot90(arr, (-self.rotation) % 4)
        return arr.copy().tolist()


D8_TRANSFORMS = tuple(DihedralTransform(k, flip) for k in range(4) for flip in (False, True))


def apply_color_permutation(grid: Sequence[Sequence[int]], perm: Mapping[int, int]) -> list[list[int]]:
    """Apply a color mapping while preserving values not present in ``perm``."""

    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    for src, dst in perm.items():
        out[arr == src] = dst
    return out.tolist()


def random_color_permutation(rng: random.Random | None = None, keep_background: bool = True) -> dict[int, int]:
    """Return a random ARC color permutation."""

    rng = rng or random
    colors = list(range(10))
    fixed = [0] if keep_background else []
    movable = [c for c in colors if c not in fixed]
    shuffled = movable[:]
    rng.shuffle(shuffled)
    perm = {c: c for c in fixed}
    perm.update(dict(zip(movable, shuffled)))
    return perm


def sample_color_permutations(
    count: int,
    rng: random.Random | None = None,
    keep_background: bool = True,
) -> list[dict[int, int]]:
    """Sample unique color permutations, including identity first."""

    rng = rng or random
    identity = {c: c for c in range(10)}
    if count <= 1:
        return [identity]

    perms = [identity]
    seen = {tuple(identity[i] for i in range(10))}
    attempts = 0
    while len(perms) < count and attempts < count * 50:
        perm = random_color_permutation(rng, keep_background=keep_background)
        key = tuple(perm[i] for i in range(10))
        if key not in seen:
            seen.add(key)
            perms.append(perm)
        attempts += 1
    return perms


def exhaustive_small_color_permutations(colors: Iterable[int], limit: int = 120) -> list[dict[int, int]]:
    """Generate deterministic permutations for the active non-background colors."""

    active = sorted(c for c in set(colors) if c != 0)
    perms: list[dict[int, int]] = []
    for shuffled in itertools.islice(itertools.permutations(active), limit):
        perm = {c: c for c in range(10)}
        perm.update(dict(zip(active, shuffled)))
        perms.append(perm)
    return perms or [{c: c for c in range(10)}]


def translate_with_padding(
    grid: Sequence[Sequence[int]],
    dx: int,
    dy: int,
    pad_value: int = 0,
) -> list[list[int]]:
    """Shift a grid inside its original canvas and pad uncovered cells."""

    arr = np.asarray(grid, dtype=np.int64)
    h, w = arr.shape
    out = np.full_like(arr, pad_value)
    src_r0, src_r1 = max(0, -dy), min(h, h - dy)
    src_c0, src_c1 = max(0, -dx), min(w, w - dx)
    dst_r0, dst_r1 = max(0, dy), min(h, h + dy)
    dst_c0, dst_c1 = max(0, dx), min(w, w + dx)
    if src_r0 < src_r1 and src_c0 < src_c1:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out.tolist()


def transform_task(task: dict, transform: DihedralTransform, color_perm: Mapping[int, int] | None = None) -> dict:
    """Apply geometry and optional color permutation to all task grids."""

    def aug_grid(grid: Sequence[Sequence[int]]) -> list[list[int]]:
        out = transform.apply(grid)
        if color_perm is not None:
            out = apply_color_permutation(out, color_perm)
        return out

    return {
        **{k: v for k, v in task.items() if k not in {"train", "test"}},
        "train": [{"input": aug_grid(p["input"]), "output": aug_grid(p["output"])} for p in task["train"]],
        "test": [
            {
                "input": aug_grid(p["input"]),
                **({"output": aug_grid(p["output"])} if "output" in p else {}),
            }
            for p in task["test"]
        ],
    }
