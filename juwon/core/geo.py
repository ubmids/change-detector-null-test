"""Georeferencing: turn pixel-space detections into lon/lat and GeoJSON.

The AOI bounding box + the scene's pixel dimensions give a simple affine
(north-up, equirectangular over a small AOI — accurate enough at these scales).
This is what makes a detection *intelligence* instead of a blob: a point on a
map you can plot, export, and hand to an analyst or a partner system.
"""
from __future__ import annotations

from ..models import AOI, Detection


def pixel_to_lonlat(aoi: AOI, height: int, width: int, x: float, y: float) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = aoi.bbox
    lon = min_lon + (x / width) * (max_lon - min_lon)
    lat = max_lat - (y / height) * (max_lat - min_lat)  # image y=0 is the top (max lat)
    return round(lon, 6), round(lat, 6)


def georeference(det: Detection, aoi: AOI, height: int, width: int) -> Detection:
    cx, cy = det.centroid_px
    lon, lat = pixel_to_lonlat(aoi, height, width, cx, cy)
    x0, y0, x1, y1 = det.bbox_px
    lon0, lat1 = pixel_to_lonlat(aoi, height, width, x0, y0)
    lon1, lat0 = pixel_to_lonlat(aoi, height, width, x1, y1)
    return det.model_copy(
        update={"lon": lon, "lat": lat, "bbox_lonlat": (lon0, lat0, lon1, lat1)}
    )


def detections_to_geojson(detections: list[Detection]) -> dict:
    """A FeatureCollection of detection centroids — drops straight onto a map."""
    features = []
    for i, d in enumerate(detections):
        if d.lon is None or d.lat is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [d.lon, d.lat]},
                "properties": {
                    "id": i,
                    "class": d.cls,
                    "confidence": d.confidence,
                    "area_m2": d.area_m2,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def ground_sample_m(aoi: AOI, size_px: int) -> float:
    """Actual metres per pixel when an AOI is read into a size_px x size_px grid.

    Providers resample to a fixed tile size, so the true resolution depends on
    how large the AOI is — NOT on the sensor's native 10 m. Reporting the
    sensor value made every m² figure wrong for any AOI bigger than ~2.5 km.
    """
    min_lon, min_lat, max_lon, max_lat = aoi.bbox
    import math

    mid_lat = math.radians((min_lat + max_lat) / 2)
    w_m = (max_lon - min_lon) * 111_320 * math.cos(mid_lat)
    h_m = (max_lat - min_lat) * 110_540
    return round(max(w_m, h_m) / max(size_px, 1), 2)


def tile_size_for(aoi: AOI, target_m: float = 10.0, lo: int = 256, hi: int = 1024) -> int:
    """Pixels per side needed to read an AOI at ~target_m ground sampling.

    A fixed tile size silently destroys resolution on larger AOIs — a 6 km box
    read at 256 px is 24 m/px, where a 30 m building is one pixel and cannot be
    seen at all. Scaling the request to the sensor's native 10 m keeps small
    structures detectable; the cap keeps big AOIs affordable (use tiling for
    those instead).
    """
    import math

    min_lon, min_lat, max_lon, max_lat = aoi.bbox
    mid_lat = math.radians((min_lat + max_lat) / 2)
    span_m = max(
        (max_lon - min_lon) * 111_320 * math.cos(mid_lat),
        (max_lat - min_lat) * 110_540,
    )
    return int(min(hi, max(lo, math.ceil(span_m / target_m))))
