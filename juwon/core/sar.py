"""SAR-specific processing: speckle filtering and water masking.

Radar is not optical, and treating it as if it were produces nonsense. Two
things must happen before the generic pipeline can be trusted on SAR:

1. **Speckle.** Coherent imaging produces multiplicative salt-and-pepper noise
   that changes on every pass. Run raw SAR through change detection and you get
   hundreds of "change regions" that are pure speckle — measured at Lagos:
   921 regions, 16% of the area, all noise. A boxcar multi-look filter is the
   standard, cheap remedy.

2. **Water is dark, not blue.** NDWI needs optical bands and returns nothing on
   SAR. But water is trivially identifiable in radar: smooth surfaces reflect
   the signal away, so open water is the darkest thing in the scene. That makes
   the maritime case *easier* on SAR than on optical — a ship is a bright,
   compact return on a near-black background.
"""
from __future__ import annotations

import numpy as np

from .imagery.base import Scene


def boxcar(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Fast mean filter via an integral image (numpy only)."""
    if size < 2:
        return img
    pad = size // 2
    p = np.pad(img.astype(np.float64), pad, mode="edge")
    integral = p.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    h, w = img.shape
    total = (integral[size:size + h, size:size + w]
             - integral[:h, size:size + w]
             - integral[size:size + h, :w]
             + integral[:h, :w])
    return (total / (size * size)).astype(img.dtype)


def despeckle(scene: Scene, size: int = 3) -> Scene:
    """Multi-look the scene to suppress speckle before comparison."""
    if not is_sar(scene):
        return scene
    data = np.stack([boxcar(scene.data[b], size) for b in range(scene.data.shape[0])])
    return Scene(data=data, band_names=scene.band_names, date=scene.date, aoi=scene.aoi,
                 source=scene.source, resolution_m=scene.resolution_m, mask=scene.mask)


def is_sar(scene: Scene) -> bool:
    return scene.has_bands("vv") or scene.has_bands("hh")


def sar_water_mask(scene: Scene, min_separation: float = 0.06,
                   min_class_frac: float = 0.05) -> np.ndarray | None:
    """Open water = the dark mode of a bimodal radar scene, or None.

    Threshold comes from the scene's own histogram (Otsu) rather than a fixed dB
    value, so it adapts to incidence angle, wind roughening and sea state.

    Crucially it **refuses to answer on a unimodal scene**. A naive
    "darkest N%" rule always finds water — on a uniformly dark scene it labelled
    94% of the image as water, which then masked away the very change we were
    looking for. If the two classes aren't meaningfully separated, or one of them
    is negligible, there is no land/water boundary here and the honest answer is
    "unknown", not a guess.
    """
    if not is_sar(scene):
        return None
    band = scene.band("vv") if scene.has_bands("vv") else scene.data[0]
    smooth = boxcar(band, 5)
    valid = smooth[~scene.mask] if scene.mask is not None else smooth.ravel()
    if valid.size < 64:
        return None

    thr = _otsu(valid)
    dark, bright = valid[valid <= thr], valid[valid > thr]
    if dark.size == 0 or bright.size == 0:
        return None
    # both classes must be substantive, and genuinely far apart
    frac = dark.size / valid.size
    if not (min_class_frac <= frac <= 1.0 - min_class_frac):
        return None
    if float(bright.mean() - dark.mean()) < min_separation:
        return None
    return smooth <= thr


def _otsu(values: np.ndarray, bins: int = 128) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        return lo
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom < 1e-9] = 1e-9
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(mids[int(np.argmax(sigma_b))])
