"""E4/E5: the null test on certified-invariant ground.

The mechanism experiment (null_sweep.py) shows what threshold selectors do to
noise. This one asks what a complete pipeline does to real imagery of ground
that somebody else has already certified as unchanging.

SITE SELECTION IS THE POINT. These are not deserts we picked because they look
empty. They are the six sites endorsed by CEOS as standard reference targets
for post-launch radiometric calibration of optical sensors, whose invariance is
independently established: Khadka, Teixeira Pinto & Leigh (2021) ran
change-point detection on the sites themselves and found a maximum trend of
0.215% in TOA reflectance per year. So the null is defended in the literature
before we touch it, which is exactly what an ad-hoc choice of "somewhere quiet"
cannot give you.

Coordinates are from the USGS EROS CalVal Center of Excellence site catalogue
(calval.cr.usgs.gov/apps/<site>), fetched 2026-09-22. Each site's page gives a
centre, a CEOS region of interest of roughly one degree, and a tighter CNES
region of interest of roughly 0.2 degrees. We take a 0.1 degree box centred in
the CNES region, because 0.2 degrees is about 20 km and would force the
analysis grid past its 1024 px ceiling, dropping the ground sample to ~20 m.
Testing at the sensor's native 10 m matters here: misreporting resolution was
one of the defects this protocol originally found.

Temporal separation is swept (E5) because nuisance grows with it and pipeline
error does not, so the short-separation intercept is the part attributable to
the pipeline rather than to the world.

Two conditions per pair:
  calibrated   the pipeline as it ships
  no_floor     min_magnitude = 0, i.e. the threshold chosen purely from the
               data, which is the condition null_sweep.py characterises

Run:  .venv/bin/python paper/field_null.py --out paper/results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from juwon.core.change.change import detect_change  # noqa: E402
from juwon.core.imagery.registry import make_provider  # noqa: E402
from juwon.models import AOI  # noqa: E402


HALF = 0.05  # degrees; a 0.1 x 0.1 box, ~10 km, so the grid holds native 10 m


def _as_date(v) -> _date | None:
    """Scene.date may be a date or an ISO-ish string, depending on provider."""
    if isinstance(v, _date):
        return v
    try:
        return _date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _sep_days(a, b) -> int | None:
    da, db = _as_date(a), _as_date(b)
    return abs((db - da).days) if da and db else None


def _box(lon: float, lat: float) -> tuple[float, float, float, float]:
    return (lon - HALF, lat - HALF, lon + HALF, lat + HALF)


# Centres of the CNES region of interest for each CEOS-endorsed PICS, from the
# USGS ECCOE catalogue. Libya-1's published centre (24.42) sits on the edge of
# its published CNES box (24.40-24.60), so we use the box centre for that one
# and note the discrepancy rather than silently picking one.
PICS = {
    "libya-4": _box(23.39, 28.55),
    "algeria-3": _box(7.66, 30.32),
    "libya-1": _box(13.30, 24.50),
    "algeria-5": _box(2.23, 31.02),
    "mauritania-1": _box(-9.30, 19.40),
    "mauritania-2": _box(-8.78, 20.85),
}

# Carried over from the original two-site calibration so the new numbers can be
# read against the old ones. Aswan is an uncertified arid control; Lagos is the
# deliberate hard case, a turbid tropical estuary, and is NOT a null site: real
# change happens there. It is included to show what the protocol reports when
# the invariance premise does not hold.
LEGACY = {
    "aswan-desert": (32.85, 24.05, 32.91, 24.10),
    "lagos-harbour": (3.38, 6.42, 3.44, 6.47),
}

WINDOWS = {
    "short": ("2024-01-01", "2024-02-15"),
    "medium": ("2024-01-01", "2024-04-01"),
    "long": ("2023-06-01", "2024-06-01"),
}


def _sites() -> dict[str, tuple[float, float, float, float]]:
    out = dict(PICS)
    out.update(LEGACY)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/results")
    ap.add_argument("--only", default=None, help="comma-separated site keys")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sites = _sites()
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        sites = {k: v for k, v in sites.items() if k in keep}

    provider = make_provider("sentinel")
    rows: list[dict] = []

    for key, bbox in sites.items():
        aoi = AOI(bbox=bbox, name=key)
        for wname, (start, end) in WINDOWS.items():
            rec: dict = {
                "site": key,
                "certified_pics": key in PICS,
                "window": wname,
                "start": start,
                "end": end,
                "bbox": list(bbox),
            }
            t0 = time.time()
            try:
                a, b = provider.get_pair(aoi, start, end)
                rec["t0_date"] = str(a.date)
                rec["t1_date"] = str(b.date)
                rec["resolution_m"] = float(getattr(a, "resolution_m", float("nan")))
                rec["separation_days"] = _sep_days(a.date, b.date)
                rec["t0_masked_frac"] = (
                    float(np.mean(~a.mask)) if getattr(a, "mask", None) is not None else 0.0
                )
                rec["t1_masked_frac"] = (
                    float(np.mean(~b.mask)) if getattr(b, "mask", None) is not None else 0.0
                )

                # identity check: a scene against itself must be exactly zero,
                # otherwise nothing below means anything
                ident, _ = detect_change(a, a)
                rec["identity_pct"] = float(ident.pct_area_changed)

                cal, _ = detect_change(a, b)
                rec["calibrated_pct"] = float(cal.pct_area_changed)
                rec["calibrated_regions"] = int(cal.num_regions)

                raw, _ = detect_change(a, b, min_magnitude=0.0)
                rec["no_floor_pct"] = float(raw.pct_area_changed)
                rec["no_floor_regions"] = int(raw.num_regions)

                rec["status"] = "ok"
            except Exception as e:  # refusals are data, not failures
                rec["status"] = "refused"
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["traceback"] = traceback.format_exc(limit=3)
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(rec)

            tag = "PICS" if rec["certified_pics"] else "    "
            if rec["status"] == "ok":
                print(
                    f"{tag} {key:<16} {wname:<7} "
                    f"{rec['t0_date']}->{rec['t1_date']} "
                    f"({str(rec['separation_days'] or '?'):>4}d)  "
                    f"calibrated {rec['calibrated_pct']:6.2f}%   "
                    f"no-floor {rec['no_floor_pct']:6.2f}%   "
                    f"identity {rec['identity_pct']:.2f}%",
                    flush=True,
                )
            else:
                print(f"{tag} {key:<16} {wname:<7} REFUSED  {rec['error'][:70]}", flush=True)

            (out / "field_null.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if r["status"] == "ok" and r["certified_pics"]]
    if ok:
        cal = [r["calibrated_pct"] for r in ok]
        nof = [r["no_floor_pct"] for r in ok]
        print("\n--- certified PICS only ---")
        print(f"n = {len(ok)} site-pairs across {len({r['site'] for r in ok})} sites")
        print(
            f"calibrated   median {np.median(cal):.2f}%  "
            f"mean {np.mean(cal):.2f}%  max {np.max(cal):.2f}%"
        )
        print(
            f"no floor     median {np.median(nof):.2f}%  "
            f"mean {np.mean(nof):.2f}%  max {np.max(nof):.2f}%"
        )
    print(f"\nWrote {out / 'field_null.json'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
