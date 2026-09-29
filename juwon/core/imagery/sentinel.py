"""Real free imagery: Sentinel-2 L2A via the Element84 Earth Search STAC API.

Search is stdlib-only (urllib). Reading the COG pixels needs `rasterio` +
`pyproj`, imported lazily so the rest of the engine runs without them. This is
the provider you point at real coordinates on your own machine:

    pip install rasterio pyproj
    JUWON_PROVIDER=sentinel python scripts/demo.py

Not exercised in the offline test suite (no network / no rasterio there); the
FakeImageryProvider covers the pipeline logic.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date

import numpy as np

from ...config import settings
from ...models import AOI
from ..geo import ground_sample_m, tile_size_for
from .base import Scene

# Sentinel-2 L2A asset keys on Earth Search for the bands we use.
_ASSETS = {"red": "red", "green": "green", "blue": "blue", "nir": "nir"}
_ORDER = ["red", "green", "blue", "nir"]

# SCL classes to treat as invalid: 0 nodata, 3 cloud shadow, 8/9 cloud
# medium/high probability, 10 thin cirrus.
_SCL_INVALID = (0, 3, 8, 9, 10)


def _stac_search(
    aoi: AOI, start: str, end: str, limit: int = 10, *,
    collection: str = "sentinel-2-l2a", cloud_query: bool = True,
) -> list[dict]:
    payload: dict = {
        "collections": [collection],
        "bbox": list(aoi.bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
        "limit": limit,
    }
    if cloud_query:
        payload["query"] = {"eo:cloud_cover": {"lt": settings.max_cloud_cover}}
    req = urllib.request.Request(
        f"{settings.stac_url}/search",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["features"]


def _search_limit(start: str, end: str) -> int:
    """How many scenes to ask STAC for, scaled to the window.

    A fixed limit silently truncates the request: the search sorts ascending,
    so asking for twelve scenes over a one-year window returns the first twelve
    and "late" then means early February rather than the following spring. The
    caller gets a fifty-day comparison while believing they asked for a year.

    Sentinel-2 revisits every ~5 days and overlapping tiles can double that, so
    budget ~2 scenes per revisit and keep a floor for short windows and a cap
    for very long ones.
    """
    from datetime import date as _d

    try:
        days = abs((_d.fromisoformat(end[:10]) - _d.fromisoformat(start[:10])).days)
    except ValueError:
        return 12
    return int(min(500, max(12, (days / 5) * 2 + 8)))


class SentinelProvider:
    name = "sentinel-2-l2a"

    def __init__(self, size: int | None = None, candidates: int = 3,
                 max_aoi_cloud: float = 0.15):
        # None = adapt per AOI to ~10 m native; an int pins it (tests/demos).
        self.size = size
        self.candidates = candidates      # scenes to try per end of the window
        self.max_aoi_cloud = max_aoi_cloud  # reject if the AOI itself is cloudier

    def _size_for(self, aoi: AOI) -> int:
        return self.size if self.size else tile_size_for(aoi)

    def _read_asset(self, href: str, aoi: AOI, *, resampling_nearest: bool = False) -> np.ndarray:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        # Sentinel-1 assets live on a public S3 bucket that GDAL will refuse to
        # touch unless we explicitly ask anonymously; harmless for HTTPS assets.
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), \
                rasterio.open(href) as ds:
            left, bottom, right, top = transform_bounds("EPSG:4326", ds.crs, *aoi.bbox)
            win = from_bounds(left, bottom, right, top, ds.transform)
            return ds.read(
                1,
                window=win,
                out_shape=(self._size_for(aoi), self._size_for(aoi)),
                boundless=True,
                fill_value=0,
                resampling=Resampling.nearest if resampling_nearest else Resampling.bilinear,
            )

    def _read_scene(self, feature: dict, aoi: AOI) -> Scene:
        # Each band is an independent windowed HTTP read of a remote COG, so
        # fetch them concurrently — this is ~5x faster than sequential and is
        # the difference between a usable UI and a two-minute wait.
        from concurrent.futures import ThreadPoolExecutor

        jobs = [(feature["assets"][_ASSETS[k]]["href"], False) for k in _ORDER]
        scl_asset = feature["assets"].get("scl")
        if scl_asset is not None:
            jobs.append((scl_asset["href"], True))

        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            arrays = list(pool.map(lambda j: self._read_asset(j[0], aoi, resampling_nearest=j[1]), jobs))

        bands = [a.astype(np.float32) / 10000.0 for a in arrays[: len(_ORDER)]]  # L2A -> reflectance
        data = np.clip(np.stack(bands), 0.0, 1.0)

        # Cloud/shadow mask from the scene classification layer (SCL, 20 m).
        mask = np.isin(arrays[-1], _SCL_INVALID) if scl_asset is not None else None

        return Scene(
            data=data,
            band_names=list(_ORDER),
            date=feature["properties"]["datetime"][:10],
            aoi=aoi,
            source="sentinel-2-l2a",
            resolution_m=ground_sample_m(aoi, self._size_for(aoi)),
            mask=mask,
        )

    def get_pair(self, aoi: AOI, start: str, end: str) -> tuple[Scene, Scene]:
        """Earliest and latest scenes that are actually clear OVER THIS AOI.

        STAC's `eo:cloud_cover` describes the whole ~110 km tile, which says
        nothing about a 6 km box inside it: at Lagos it happily returned tiles
        rated <20% cloud that were 24% and 33% cloud over the AOI itself. So we
        read several candidates, measure each one's real masked fraction from its
        own SCL band, and keep the clearest early + clearest late scene.
        """
        from concurrent.futures import ThreadPoolExecutor

        feats = _stac_search(aoi, start, end, limit=_search_limit(start, end))
        if len(feats) < 2:
            raise RuntimeError(
                f"need >=2 Sentinel-2 scenes for {aoi.bbox} in {start}..{end}, "
                f"found {len(feats)}. Widen the date range or raise JUWON_MAX_CLOUD."
            )

        # Sample from both ends of the window so we still compare across time.
        half = max(1, len(feats) // 2)
        early, late = feats[:half], feats[half:]
        cand = (early[: self.candidates] + late[-self.candidates :])

        with ThreadPoolExecutor(max_workers=min(4, len(cand))) as pool:
            scenes = list(pool.map(lambda f: self._read_scene(f, aoi), cand))

        def cloudiness(sc: Scene) -> float:
            return float(sc.mask.mean()) if sc.mask is not None else 0.0

        n_early = len(early[: self.candidates])
        early_scenes = sorted(scenes[:n_early], key=cloudiness)
        late_scenes = sorted(scenes[n_early:], key=cloudiness)
        if not early_scenes or not late_scenes:
            ordered = sorted(scenes, key=lambda s: s.date)
            return ordered[0], ordered[-1]

        t0, t1 = early_scenes[0], late_scenes[0]
        if cloudiness(t0) > self.max_aoi_cloud or cloudiness(t1) > self.max_aoi_cloud:
            raise RuntimeError(
                f"no clear-enough imagery over this area in {start}..{end}: best pair is "
                f"{100*cloudiness(t0):.0f}% and {100*cloudiness(t1):.0f}% cloud over the AOI "
                f"(limit {100*self.max_aoi_cloud:.0f}%). Try a wider date range, a drier "
                f"season, or radar (Sentinel-1), which sees through cloud."
            )
        return t0, t1

    def get_scene(self, aoi: AOI, date: str) -> Scene:
        """Nearest low-cloud scene on/after `date` (±30-day window)."""
        from datetime import date as _date, timedelta

        d = _date.fromisoformat(date)
        feats = _stac_search(aoi, date, (d + timedelta(days=30)).isoformat(), limit=1)
        if not feats:
            raise RuntimeError(f"no Sentinel-2 scene near {date} for {aoi.bbox}")
        return self._read_scene(feats[0], aoi)
