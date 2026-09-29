"""E6: the other half of the null test.

A detector that reports nothing passes a null test perfectly, so the null test
alone is not a validation, it is half of one. This is the other half: paint a
structure of known real-world size into the genuine second acquisition and ask
whether it survives the same thresholds that produced the null.

The pair is the point. Null sites give a false-alarm rate with no labels;
injection gives a detection floor with no labels. Together they are both axes
of a calibration curve, obtained without anyone annotating anything.

Runs against the same cached PICS pairs as field_null.py, so run that first.

Run:  .venv/bin/python paper/injection.py --out paper/results
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for field_null

import numpy as np  # noqa: E402

from juwon.core.change.change import detect_change  # noqa: E402
from juwon.core.imagery.base import Scene  # noqa: E402
from juwon.core.imagery.registry import make_provider  # noqa: E402
from juwon.models import AOI  # noqa: E402

from field_null import PICS, WINDOWS  # noqa: E402

# Sizes bracket the configured reporting floor (2500 m2, about a 50 m
# structure) and the previously measured detection edge (60 m found, 30 m not).
SIZES_M = [20, 30, 40, 50, 60, 80, 120, 200]

# Contrast added to reflectance. 0.45 is what the original harness used; the
# lower values ask how much of the detection floor is size and how much is
# brightness, which the original could not separate.
CONTRASTS = [0.45, 0.20, 0.10]


def inject(scene: Scene, size_m: float, contrast: float) -> Scene:
    """Paint a bright square of known ground size into a real scene.

    Size is converted through the scene's own reported ground sample, so if
    that number is wrong the injected structure is the wrong size, which is
    precisely the failure mode that makes resolution misreporting dangerous.
    """
    px = max(1, int(round(size_m / scene.resolution_m)))
    data = scene.data.copy()
    _, h, w = data.shape
    y, x = (h - px) // 2, (w - px) // 2
    data[:, y : y + px, x : x + px] = np.clip(
        data[:, y : y + px, x : x + px] + contrast, 0, 1
    )
    footprint = np.zeros((h, w), dtype=bool)
    footprint[y : y + px, x : x + px] = True
    return footprint, Scene(
        data=data,
        band_names=scene.band_names,
        date=scene.date,
        aoi=scene.aoi,
        source=scene.source,
        resolution_m=scene.resolution_m,
        mask=scene.mask,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/results")
    ap.add_argument("--window", default="short", choices=list(WINDOWS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start, end = WINDOWS[args.window]

    provider = make_provider("sentinel")
    rows: list[dict] = []

    for key, bbox in PICS.items():
        aoi = AOI(bbox=bbox, name=key)
        try:
            t0, t1 = provider.get_pair(aoi, start, end)
        except Exception as e:
            print(f"{key:<16} unavailable: {type(e).__name__}: {e}", flush=True)
            rows.append({"site": key, "status": "refused", "error": str(e)})
            continue

        base, _ = detect_change(t0, t1)
        gsd = float(t1.resolution_m)
        print(f"\n{key}  ({t0.date} -> {t1.date}, {gsd:.2f} m/px)")
        print(f"  baseline, nothing injected: {base.pct_area_changed:.2f}% "
              f"/ {base.num_regions} regions")

        for contrast in CONTRASTS:
            found_at = None
            line = []
            for size in SIZES_M:
                try:
                    footprint, inj = inject(t1, size, contrast)
                    st, mask = detect_change(t0, inj)
                    footprint_px = int(footprint.sum())
                    total_px = t1.data.shape[1] * t1.data.shape[2]
                    expected_pct = 100.0 * footprint_px / total_px
                    gain = st.pct_area_changed - base.pct_area_changed
                    # detected = the change mask actually covers the structure
                    hit = float(mask[footprint].mean()) if footprint_px else 0.0
                    detected = hit >= 0.5
                    if detected and found_at is None:
                        found_at = size
                    line.append(f"{size}m:{'Y' if detected else '.'}")
                    rows.append(
                        {
                            "site": key,
                            "status": "ok",
                            "window": args.window,
                            "t0": str(t0.date),
                            "t1": str(t1.date),
                            "gsd_m": gsd,
                            "contrast": contrast,
                            "size_m": size,
                            "area_m2": size * size,
                            "baseline_pct": float(base.pct_area_changed),
                            "injected_pct": float(st.pct_area_changed),
                            "expected_pct": float(expected_pct),
                            "gain_pct": float(gain),
                            "footprint_recall": hit,
                            "area_collapsed": bool(gain < -0.1),
                            "detected": bool(detected),
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "site": key,
                            "status": "error",
                            "contrast": contrast,
                            "size_m": size,
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(limit=2),
                        }
                    )
                    line.append(f"{size}m:!")
            print(f"  +{contrast:<5.2f} reflectance   " + "  ".join(line)
                  + f"   -> floor {found_at or '>200'} m")
            (out / "injection.json").write_text(json.dumps(rows, indent=2))

    print(f"\nWrote {out / 'injection.json'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
