"""ARC grid visualization helpers."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ARC_COLORS = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]


def plot_grid(grid, ax=None, title: str | None = None):
    ax = ax or plt.gca()
    cmap = mcolors.ListedColormap(ARC_COLORS)
    ax.imshow(np.asarray(grid), cmap=cmap, vmin=0, vmax=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
