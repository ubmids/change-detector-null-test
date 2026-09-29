"""Change detection between two aligned scenes.

Pipeline:
  1. Relative radiometric normalization — fit t1 onto t0's brightness scale per
     band (robust, inlier-refit) so global illumination/atmospheric/seasonal
     differences don't masquerade as change. This is the single biggest driver
     of false positives, so it's on by default.
  2. Spectral change magnitude — L2 distance across all bands (richer than a
     single-band brightness diff).
  3. Otsu threshold — data-driven split of "changed" vs "unchanged" instead of
     a hand-tuned constant.
  4. Region grouping + NDVI vegetation delta.
"""
from __future__ import annotations

import numpy as np

from ...models import ChangeStats
from ..cc import label_components, region_props
from ..imagery.base import Scene
from ..sar import despeckle, is_sar, sar_water_mask

_EPS = 1e-6


def ndvi(scene: Scene) -> np.ndarray:
    nir, red = scene.band("nir"), scene.band("red")
    return (nir - red) / (nir + red + _EPS)


def _robust_normalize(t0: np.ndarray, t1: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Return t1 rescaled per-band onto t0 via a robust linear fit (y=a·x+b),
    so unchanged ground matches and only genuine change survives as residual.

    The fit must be computed on VALID pixels only. Fitting over everything lets
    a harbour's dominant water surface (or a cloud bank) drag the coefficients
    and mis-scale the land — measured as a regression from 2% to 14.5% false
    change at Lagos before this was fixed.
    """
    bands = t0.shape[0]
    out = np.empty_like(t1)
    for b in range(bands):
        x_all, y_all = t0[b].ravel(), t1[b].ravel()
        if valid is not None and valid.any():
            v = valid.ravel()
            x, y = x_all[v], y_all[v]
        else:
            x, y = x_all, y_all
        if x.size < 16 or x.std() < _EPS:  # too little to fit against
            out[b] = t1[b]
            continue
        a, off = np.polyfit(x, y, 1)
        resid = y - (a * x + off)
        std = resid.std()
        if std > _EPS:
            inliers = np.abs(resid) <= 2.0 * std  # drop the changed pixels, refit
            if inliers.sum() > 10:
                a, off = np.polyfit(x[inliers], y[inliers], 1)
        out[b] = (t1[b] - off) / a if abs(a) > _EPS else t1[b]
    return out


def _otsu(values: np.ndarray, bins: int = 128) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hi = float(v.max())
    if hi <= _EPS:
        return 0.0
    hist, edges = np.histogram(v, bins=bins, range=(0.0, hi))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom < _EPS] = _EPS
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(mids[int(np.argmax(sigma_b))])


def _ndwi(scene: Scene) -> np.ndarray:
    green, nir = scene.band("green"), scene.band("nir")
    return (green - nir) / (green + nir + _EPS)


def water_mask(t0: Scene, t1: Scene) -> np.ndarray | None:
    """Pixels that are water in EITHER scene — NDWI on optical, darkness on SAR.

    A harbour's surface changes on every tide and every wave; treating that as
    "change" is what produced the bogus 38.9% at Lagos. Masking water lets real
    events on land — new quays, cleared ground, new structures — stand out.
    Vessels are handled by the detector, not by change detection.
    """
    if is_sar(t0) and is_sar(t1):
        m0, m1 = sar_water_mask(t0), sar_water_mask(t1)
        return None if (m0 is None or m1 is None) else (m0 | m1)
    if not (t0.has_bands("green", "nir") and t1.has_bands("green", "nir")):
        return None
    return (_ndwi(t0) > 0.2) | (_ndwi(t1) > 0.2)


def detect_change(
    t0: Scene,
    t1: Scene,
    *,
    min_region_m2: float = 2500.0,  # ~a 50 m structure — "worth telling a human"
    min_magnitude: float = 0.10,    # absolute floor on spectral distance
    veg_delta: float = 0.2,
    normalize: bool = True,
    cloud_mask: np.ndarray | None = None,
    mask_water: bool = True,        # water surface change is tide/waves, not events
    min_region_px: int | None = None,  # explicit override, else derived from m²
) -> tuple[ChangeStats, np.ndarray]:
    if t0.shape != t1.shape:
        raise ValueError(f"scenes must be aligned; got {t0.shape} vs {t1.shape}")

    # Speckle is multiplicative noise that re-randomises every pass; comparing
    # raw SAR yields hundreds of phantom regions (measured: 921 at Lagos).
    sar = is_sar(t0) and is_sar(t1)
    if sar:
        t0, t1 = despeckle(t0, 5), despeckle(t1, 5)
        # Residual speckle sits well above optical sensor noise, so the absolute
        # floor has to rise with it or every pass "changes".
        min_magnitude = max(min_magnitude, 0.22)

    # Invalid = clouds/shadows/nodata (+ optionally water, whose surface changes
    # constantly and meaninglessly). Computed first: the normalization fit and
    # the threshold must both ignore these pixels.
    invalid = None
    extra = water_mask(t0, t1) if mask_water else None
    for m in (t0.mask, t1.mask, cloud_mask, extra):
        if m is not None:
            invalid = m if invalid is None else (invalid | m)
    valid = ~invalid if invalid is not None else None

    d0 = t0.data

    def magnitude(d1: np.ndarray) -> np.ndarray:
        m = np.sqrt(np.sum((d1 - d0) ** 2, axis=0))
        return np.where(invalid, 0.0, m) if invalid is not None else m

    mag = magnitude(t1.data)
    if normalize:
        # Normalization helps on stable land but can hurt where the scene-to-
        # scene relationship is unstable (turbid water, haze): measured 20%->7%
        # false change at Aswan, but 2%->14.5% WORSE at Lagos. So don't trust it
        # blindly — keep whichever version leaves less background noise. Real
        # change is sparse, so the median magnitude is a clean proxy for noise.
        cand = magnitude(_robust_normalize(d0, t1.data, valid))
        sel = valid if valid is not None else np.ones(mag.shape, dtype=bool)
        if sel.any() and np.median(cand[sel]) < np.median(mag[sel]):
            mag = cand

    # Otsu is a RELATIVE split: on a static scene it still "finds" a threshold
    # and reports pure sensor noise as change (measured: 7% of a static desert).
    # Taking the max against an absolute floor is what makes a null test null.
    sample = mag[valid] if valid is not None else mag
    thr = max(_otsu(sample), min_magnitude)
    mask = mag > thr
    if invalid is not None:
        mask &= ~invalid

    # Region size in real units, so the same setting means the same thing at
    # 10 m Sentinel and at 30 cm commercial resolution.
    if min_region_px is None:
        min_region_px = max(4, int(round(min_region_m2 / (t1.resolution_m ** 2))))

    labels, count = label_components(mask)
    regions = region_props(labels, count, min_area=min_region_px)

    clean = np.zeros_like(mask)
    largest = None
    for r in regions:
        clean[labels == r.label] = True
        if largest is None or r.area > largest.area:
            largest = r

    # NDVI needs optical bands — SAR scenes (vv/vh) skip vegetation stats.
    if t0.has_bands("red", "nir") and t1.has_bands("red", "nir"):
        ndc = ndvi(t1) - ndvi(t0)
        veg_lost = round(float((ndc < -veg_delta).mean() * 100.0), 3)
        veg_gained = round(float((ndc > veg_delta).mean() * 100.0), 3)
    else:
        veg_lost = veg_gained = 0.0

    stats = ChangeStats(
        pct_area_changed=round(float(clean.mean() * 100.0), 3),
        mean_abs_change=round(float(mag.mean()), 4),
        num_regions=len(regions),
        largest_region_area_px=largest.area if largest else 0,
        largest_region_bbox_px=largest.bbox if largest else None,
        veg_lost_pct=veg_lost,
        veg_gained_pct=veg_gained,
    )
    return stats, clean
