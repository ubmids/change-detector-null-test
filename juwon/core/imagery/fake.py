"""Deterministic synthetic imagery so the whole engine runs offline.

The fake scenes tell a story on purpose: between t0 and t1 a new bright
structure appears (for change detection to catch) and the number of bright
"objects" on the dark background grows (for the detector to count). Seeded off
the AOI + date, so a given place/date is always identical — important for
reproducible tests and demos.
"""
from __future__ import annotations

import numpy as np

from ...models import AOI
from .base import Scene

BANDS = ["red", "green", "blue", "nir"]


def _rng(aoi: AOI, date: str, salt: int = 0) -> np.random.Generator:
    seed = (hash((aoi.id, date, salt)) & 0xFFFFFFFF)
    return np.random.default_rng(seed)


def _blank(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Dark, water-like background with mild speckle across 4 bands."""
    base = rng.normal(0.08, 0.02, size=(4, h, w)).astype(np.float32)
    return np.clip(base, 0.0, 1.0)


def _add_object(img: np.ndarray, rng: np.random.Generator, size: int = 3) -> None:
    """Drop a small bright blob (a ship / vehicle proxy) at a random spot."""
    _, h, w = img.shape
    y = int(rng.integers(size, h - size))
    x = int(rng.integers(size, w - size))
    img[:, y - size : y + size, x - size : x + size] += rng.uniform(0.5, 0.8)


def _add_structure(img: np.ndarray, rng: np.random.Generator) -> None:
    """Add a larger bright rectangle — a 'new building' for change detection."""
    _, h, w = img.shape
    bh, bw = int(rng.integers(10, 18)), int(rng.integers(12, 22))
    y = int(rng.integers(0, h - bh))
    x = int(rng.integers(0, w - bw))
    img[:, y : y + bh, x : x + bw] += 0.55


class FakeImageryProvider:
    """Runs with zero network / zero model weights. Great for CI and demos."""

    name = "fake"

    def __init__(self, size: int = 200, n_objects_t0: int = 4, n_objects_t1: int = 7):
        self.size = size
        self.n0 = n_objects_t0
        self.n1 = n_objects_t1

    def _scene(self, aoi: AOI, date: str, n_objects: int, new_structure: bool) -> Scene:
        rng = _rng(aoi, date)
        img = _blank(self.size, self.size, rng)
        for _ in range(n_objects):
            _add_object(img, rng)
        if new_structure:
            _add_structure(img, _rng(aoi, date, salt=99))  # fixed spot across runs
        img = np.clip(img, 0.0, 1.0)
        return Scene(
            data=img,
            band_names=list(BANDS),
            date=date,
            aoi=aoi,
            source="fake",
            resolution_m=10.0,
        )

    def get_pair(self, aoi: AOI, start: str, end: str) -> tuple[Scene, Scene]:
        t0 = self._scene(aoi, start, self.n0, new_structure=False)
        t1 = self._scene(aoi, end, self.n1, new_structure=True)
        return t0, t1

    def get_scene(self, aoi: AOI, date: str) -> Scene:
        """A single scene for time-series use; object count varies by date."""
        rng = _rng(aoi, date, salt=7)
        n = int(rng.integers(self.n0, self.n1 + 1))
        return self._scene(aoi, date, n, new_structure=False)
