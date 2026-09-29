"""Connected-components labelling and region props in pure numpy.

Kept dependency-free (no scipy/opencv) so the core engine runs anywhere. Uses
4-connectivity via iterative flood fill — fine for the tile sizes the engine
works on (a few hundred px per side).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Region:
    label: int
    area: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive-exclusive)
    centroid: tuple[float, float]  # (x, y)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected True regions. Returns (labels, count)."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for i in range(h):
        for j in range(w):
            if mask[i, j] and labels[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labels[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            stack.append((ny, nx))
    return labels, cur


def region_props(labels: np.ndarray, count: int, min_area: int = 1) -> list[Region]:
    """Compute area / bbox / centroid per label, dropping regions below min_area."""
    out: list[Region] = []
    for lab in range(1, count + 1):
        ys, xs = np.where(labels == lab)
        area = int(xs.size)
        if area < min_area:
            continue
        out.append(
            Region(
                label=lab,
                area=area,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                centroid=(float(xs.mean()), float(ys.mean())),
            )
        )
    return out
