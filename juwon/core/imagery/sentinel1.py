"""Sentinel-1 SAR — sees through cloud and at night.

Radar backscatter instead of reflectance. Ships are unambiguous bright returns
on near-black water, and cloud simply does not exist as a concept, which is why
maritime domain awareness runs on SAR rather than optical.

Source: Microsoft Planetary Computer's **RTC** (radiometrically terrain
corrected) collection, not raw GRD. That choice is load-bearing: GRD products
are stored in *radar geometry* with ground control points and no CRS, so a
window read by lat/lon is impossible — it fails with `CRS is invalid: None`.
RTC is map-projected, so it behaves exactly like Sentinel-2 and every
downstream component works unchanged.

Access is free and anonymous; Planetary Computer hands out short-lived SAS
tokens without registration, which we fetch and cache per collection.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date as _date, datetime, timedelta, timezone

import numpy as np

from ...models import AOI
from ..geo import ground_sample_m
from .base import Scene
from .sentinel import SentinelProvider

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
_BANDS = ["vv", "vh"]

_token_cache: dict[str, tuple[str, datetime]] = {}


def _sas_token(collection: str = "sentinel-1-rtc") -> str:
    """Short-lived anonymous SAS token, cached until shortly before expiry."""
    cached = _token_cache.get(collection)
    now = datetime.now(timezone.utc)
    if cached and cached[1] > now + timedelta(minutes=2):
        return cached[0]
    with urllib.request.urlopen(PC_SAS.format(collection=collection), timeout=30) as r:
        d = json.loads(r.read())
    token = d["token"]
    expiry = datetime.fromisoformat(d["msft:expiry"].replace("Z", "+00:00"))
    _token_cache[collection] = (token, expiry)
    return token


def _pc_search(aoi: AOI, start: str, end: str, limit: int = 12,
               collection: str = "sentinel-1-rtc") -> list[dict]:
    body = json.dumps({
        "collections": [collection],
        "bbox": list(aoi.bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": limit,
    }).encode()
    req = urllib.request.Request(PC_STAC, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        feats = json.loads(r.read())["features"]
    return sorted(feats, key=lambda f: f["properties"]["datetime"])


class Sentinel1Provider(SentinelProvider):
    name = "sentinel-1-rtc"
    collection = "sentinel-1-rtc"

    def _sign(self, href: str) -> str:
        return f"{href}?{_sas_token(self.collection)}"

    def _read_scene(self, feature: dict, aoi: AOI) -> Scene:  # type: ignore[override]
        from concurrent.futures import ThreadPoolExecutor

        assets = [(b, feature["assets"][b]["href"]) for b in _BANDS if b in feature["assets"]]
        if not assets:
            raise RuntimeError("Sentinel-1 scene carries no vv/vh assets")

        with ThreadPoolExecutor(max_workers=len(assets)) as pool:
            raw = list(pool.map(lambda a: self._read_asset(self._sign(a[1]), aoi), assets))

        # RTC values are linear backscatter (gamma0) spanning several orders of
        # magnitude. dB is the standard display space and is what makes the
        # dark-water / bright-ship contrast SAR is prized for actually visible.
        bands = []
        for arr in raw:
            a = np.asarray(arr, dtype=np.float32)
            db = 10.0 * np.log10(np.clip(a, 1e-5, None))      # typically -30..+5 dB
            bands.append(np.clip((db + 30.0) / 35.0, 0.0, 1.0))

        # Nodata (outside the swath) reads as exactly 0 in linear space.
        stack = np.stack([np.asarray(a) <= 0 for a in raw])
        mask = np.all(stack, axis=0)

        return Scene(
            data=np.stack(bands),
            band_names=[b for b, _ in assets],
            date=feature["properties"]["datetime"][:10],
            aoi=aoi,
            source="sentinel-1-rtc",
            resolution_m=ground_sample_m(aoi, self._size_for(aoi)),
            mask=mask if mask.any() else None,
        )

    def get_pair(self, aoi: AOI, start: str, end: str) -> tuple[Scene, Scene]:
        from concurrent.futures import ThreadPoolExecutor

        feats = _pc_search(aoi, start, end, collection=self.collection)
        if len(feats) < 2:
            raise RuntimeError(
                f"need >=2 Sentinel-1 RTC scenes for {aoi.bbox} in {start}..{end}, "
                f"found {len(feats)}. Widen the date range."
            )
        with ThreadPoolExecutor(max_workers=2) as pool:
            t0, t1 = pool.map(lambda f: self._read_scene(f, aoi), (feats[0], feats[-1]))
        return t0, t1

    def get_scene(self, aoi: AOI, date: str) -> Scene:
        d = _date.fromisoformat(date)
        feats = _pc_search(aoi, date, (d + timedelta(days=30)).isoformat(),
                           limit=1, collection=self.collection)
        if not feats:
            raise RuntimeError(f"no Sentinel-1 RTC scene near {date} for {aoi.bbox}")
        return self._read_scene(feats[0], aoi)
