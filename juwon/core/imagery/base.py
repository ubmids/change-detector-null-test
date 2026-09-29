"""Imagery abstraction: a Scene (aligned band stack) and a Provider protocol.

The rest of the engine only ever sees `Scene` objects, so swapping the free
Sentinel source for paid Planet/Maxar — or a defense partner's on-prem feed —
is a matter of adding one Provider. Nothing downstream changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ...models import AOI


@dataclass
class Scene:
    data: np.ndarray            # (bands, H, W), float32 reflectance ~[0, 1]
    band_names: list[str]       # e.g. ["red", "green", "blue", "nir"] or ["vv", "vh"]
    date: str                   # ISO date of capture
    aoi: AOI
    source: str                 # "fake", "sentinel-2-l2a", "sentinel-1-grd", ...
    resolution_m: float
    mask: np.ndarray | None = None  # (H, W) bool, True = invalid (cloud/shadow/nodata)

    def band(self, name: str) -> np.ndarray:
        return self.data[self.band_names.index(name)]

    def has_bands(self, *names: str) -> bool:
        return all(n in self.band_names for n in names)

    def brightness(self) -> np.ndarray:
        """Mean reflectance across bands — a simple panchromatic proxy."""
        return self.data.mean(axis=0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1], self.data.shape[2]


@runtime_checkable
class ImageryProvider(Protocol):
    name: str

    def get_pair(self, aoi: AOI, start: str, end: str) -> tuple[Scene, Scene]:
        """Return two aligned scenes (earliest, latest) spanning [start, end]."""
        ...
