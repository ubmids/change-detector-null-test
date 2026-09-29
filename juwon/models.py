"""Typed data models for the engine's inputs and outputs (pydantic v2).

Deliberately free of numpy arrays — every result here is JSON-serialisable so
it can cross the API boundary and land in the flywheel store unchanged.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AOI(BaseModel):
    """Area of interest as a lon/lat bounding box (WGS84)."""

    bbox: tuple[float, float, float, float] = Field(
        ..., description="(min_lon, min_lat, max_lon, max_lat)"
    )
    name: Optional[str] = None

    @field_validator("bbox")
    @classmethod
    def _valid_bbox(cls, v: tuple[float, float, float, float]):
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError("longitudes must satisfy -180 <= min_lon < max_lon <= 180")
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError("latitudes must satisfy -90 <= min_lat < max_lat <= 90")
        return v

    @property
    def id(self) -> str:
        """Stable short id for keying flywheel records to a place."""
        raw = ",".join(f"{c:.5f}" for c in self.bbox)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    @property
    def center(self) -> tuple[float, float]:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)


class Detection(BaseModel):
    cls: str = "object"
    confidence: float
    bbox_px: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel space
    centroid_px: tuple[float, float]
    area_px: int
    area_m2: Optional[float] = None
    # Georeferenced position (filled once the AOI's affine is known).
    lon: Optional[float] = None
    lat: Optional[float] = None
    bbox_lonlat: Optional[tuple[float, float, float, float]] = None
    # Feature vector that produced this detection. Persisted with the result so
    # a human label can later be turned into a training row without re-reading
    # the imagery — this is what makes the flywheel cheap.
    features: Optional[list[float]] = None
    detector: Optional[str] = None
    # Set when this scan can be compared with a previous one of the same place:
    # "arrived" (new since last time) or "stayed" (present on both dates).
    status: Optional[str] = None


class ChangeStats(BaseModel):
    pct_area_changed: float
    mean_abs_change: float
    num_regions: int
    largest_region_area_px: int = 0
    largest_region_bbox_px: Optional[tuple[int, int, int, int]] = None
    largest_region_bbox_lonlat: Optional[tuple[float, float, float, float]] = None
    veg_lost_pct: float = 0.0
    veg_gained_pct: float = 0.0


class AnalysisResult(BaseModel):
    id: str
    aoi: AOI
    t0_date: str
    t1_date: str
    source: str
    resolution_m: float
    change: ChangeStats
    detections: list[Detection] = []
    counts: dict[str, int] = {}
    summary: str = ""
    created_at: Optional[str] = None
    artifacts: dict[str, str] = {}  # name -> file path (change mask, overlay, ...)
    # Plain-language answers derived from the numbers above. This is what a
    # person actually reads; the measurements are the evidence behind it.
    findings: list[dict] = []
    headline: str = ""
    # Objects present last time and gone now. Kept so the map can show what
    # left, not only what is there — absence is often the story.
    departed: list[Detection] = []


class SeriesPoint(BaseModel):
    date: str
    count: int
    z_score: float = 0.0
    is_anomaly: bool = False


class SeriesResult(BaseModel):
    aoi: AOI
    cls: str
    points: list[SeriesPoint]
    mean_count: float
    trend_slope: float  # objects/step; >0 = growing activity
    summary: str = ""


class EvalReport(BaseModel):
    detector: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    n_analyses: int = 0


class Correction(BaseModel):
    """A human label on top of a result — the raw material for fine-tuning."""

    detection_index: Optional[int] = None  # which detection this refers to
    correct_cls: Optional[str] = None
    is_true_positive: Optional[bool] = None
    added_bbox_px: Optional[tuple[int, int, int, int]] = None  # a missed object
    note: Optional[str] = None
