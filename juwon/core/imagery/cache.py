"""Disk cache for provider scenes.

Satellite reads are the slow, rate-limited part of the pipeline (windowed COG
reads over HTTP). Caching by (provider, AOI, date-key) makes repeat analyses,
watchlist re-checks, and time series over the same site near-instant, and it
means a flaky connection can't break a demo twice. Stored as .npz.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...models import AOI
from .base import ImageryProvider, Scene


def _key(provider: str, aoi: AOI, *parts: str) -> str:
    raw = "|".join([provider, aoi.id, *parts])
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


class CachingProvider:
    """Wraps any ImageryProvider with a transparent .npz disk cache."""

    def __init__(self, inner: ImageryProvider, cache_dir: str | Path):
        self.inner = inner
        self.name = f"{inner.name}+cache"
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- persistence -----------------------------------------------------
    def _save(self, key: str, scene: Scene) -> None:
        np.savez_compressed(
            self.dir / f"{key}.npz",
            data=scene.data,
            band_names=np.array(scene.band_names),
            date=np.array(scene.date),
            source=np.array(scene.source),
            resolution_m=np.array(scene.resolution_m),
            mask=scene.mask if scene.mask is not None else np.zeros((0,), dtype=bool),
        )

    def _load(self, key: str, aoi: AOI) -> Scene | None:
        path = self.dir / f"{key}.npz"
        if not path.exists():
            return None
        z = np.load(path, allow_pickle=False)
        mask = z["mask"]
        return Scene(
            data=z["data"],
            band_names=[str(b) for b in z["band_names"]],
            date=str(z["date"]),
            aoi=aoi,
            source=str(z["source"]),
            resolution_m=float(z["resolution_m"]),
            mask=mask if mask.size else None,
        )

    # ---- provider interface ----------------------------------------------
    def get_pair(self, aoi: AOI, start: str, end: str) -> tuple[Scene, Scene]:
        k0 = _key(self.inner.name, aoi, "pair0", start, end)
        k1 = _key(self.inner.name, aoi, "pair1", start, end)
        c0, c1 = self._load(k0, aoi), self._load(k1, aoi)
        if c0 is not None and c1 is not None:
            return c0, c1
        t0, t1 = self.inner.get_pair(aoi, start, end)
        self._save(k0, t0)
        self._save(k1, t1)
        return t0, t1

    def get_scene(self, aoi: AOI, date: str) -> Scene:
        k = _key(self.inner.name, aoi, "scene", date)
        cached = self._load(k, aoi)
        if cached is not None:
            return cached
        scene = self.inner.get_scene(aoi, date)  # type: ignore[attr-defined]
        self._save(k, scene)
        return scene
