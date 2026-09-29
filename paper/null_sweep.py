"""
What does an automatic threshold find when there is nothing there?

A threshold selector is handed an image and asked where to cut. The standard
selectors choose that cut from the data itself: Otsu maximises between-class
variance, Ridler-Calvard iterates to the midpoint of the two class means, Yen
and Kapur optimise an entropy, Rosin's triangle takes the maximum deviation
from a chord. All of them are relative criteria, and a relative criterion
always has a best answer, including on an image where nothing happened.

Otsu says as much in the 1979 paper. He restricts the search to the range
where both classes are non-empty, calls the degenerate case "of course, not
our concern", and observes that the maximum therefore always exists. The
consequence is rarely quantified: handed pure sensor noise, a selector does
not decline, it partitions the noise and reports a fraction of the image as
foreground.

Rosin & Ioannidis (Pattern Recognition Letters, 2003) measured this once, on
video frames in which nothing moved, and found Otsu correctly classified
82.6% of a no-change frame, i.e. it called roughly a sixth of an empty scene
"change". As far as we can find, that single table is the only published
measurement of the effect, and nobody has swept it across selectors, noise
models and signal strength.

This script does that sweep.

  Experiment 1  signal-free input. The correct answer is zero. Report what
                each selector actually returns, across noise models.
  Experiment 2  a signal is introduced and its contrast raised from zero.
                Report the false-positive fraction on true-background pixels
                as the signal appears, which shows whether a selector ever
                had a null to return.
  Experiment 3  the fix. max(selector, floor), where the floor is a Currie
                critical level derived from the blank rather than from the
                image being thresholded.

Reference implementations come from scikit-image so that no claim here rests
on our own version of a published method. Kapur is implemented locally
because scikit-image does not ship it.

Usage:
    python paper/null_sweep.py --out paper/results
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

SIZE = 256
SEEDS = 30
NBINS = 256


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------


def threshold_kapur(image: np.ndarray, nbins: int = NBINS) -> float:
    """Kapur, Sahoo & Wong (1985): maximise the sum of the two class entropies.

    Not in scikit-image, so implemented here against the published criterion.
    """
    counts, edges = np.histogram(image.ravel(), bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = counts.astype(np.float64)
    total = p.sum()
    if total == 0:
        return float(centers[0])
    p /= total

    cum = np.cumsum(p)
    # entropy of the below-threshold class, cumulative
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    cum_plogp = np.cumsum(plogp)
    total_plogp = cum_plogp[-1]

    best_t, best_val = 0, -np.inf
    for t in range(nbins - 1):
        w0 = cum[t]
        w1 = 1.0 - w0
        if w0 <= 0 or w1 <= 0:
            continue
        h0 = np.log(w0) - cum_plogp[t] / w0
        h1 = np.log(w1) - (total_plogp - cum_plogp[t]) / w1
        val = h0 + h1
        if val > best_val:
            best_val, best_t = val, t
    return float(centers[best_t])


def _safe(fn):
    """Selectors occasionally raise on degenerate histograms; record that."""

    def wrapped(image: np.ndarray) -> float:
        try:
            return float(fn(image))
        except Exception:
            return float("nan")

    return wrapped


SELECTORS = {
    "otsu": _safe(lambda im: threshold_otsu(im, nbins=NBINS)),
    "ridler_calvard": _safe(lambda im: threshold_isodata(im, nbins=NBINS)),
    "kapur": _safe(threshold_kapur),
    "yen": _safe(lambda im: threshold_yen(im, nbins=NBINS)),
    "li": _safe(threshold_li),
    "triangle_rosin": _safe(lambda im: threshold_triangle(im, nbins=NBINS)),
    "mean": _safe(threshold_mean),
}

# Every selector above chooses its cut from the image in front of it. The
# reference below does not: it is a fixed multiple of the noise scale of a
# blank, which is what Currie (1968) calls a critical level. It is included to
# show the difference in kind rather than to compete on sensitivity.
CURRIE_K = 3.0


# --------------------------------------------------------------------------
# noise models
# --------------------------------------------------------------------------


def noise_rayleigh(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    """Magnitude of 2-D Gaussian noise.

    This is the relevant model for change magnitude: a per-pixel spectral
    difference is the length of a difference vector, so its null distribution
    is Rayleigh rather than Gaussian, and is strictly positive.
    """
    a = rng.normal(0.0, sigma, (size, size))
    b = rng.normal(0.0, sigma, (size, size))
    return np.sqrt(a * a + b * b)


def noise_halfnormal(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    """Absolute difference of a single band."""
    return np.abs(rng.normal(0.0, sigma, (size, size)))


def noise_gaussian(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    """Signed difference, included for generality."""
    return rng.normal(0.0, sigma, (size, size))


def noise_lognormal(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    """Heavy-tailed positive noise, closer to SAR speckle in character."""
    return rng.lognormal(mean=np.log(sigma), sigma=0.6, size=(size, size))


NOISE = {
    "rayleigh": noise_rayleigh,
    "halfnormal": noise_halfnormal,
    "gaussian": noise_gaussian,
    "lognormal": noise_lognormal,
}


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------


@dataclass
class Row:
    experiment: str
    noise: str
    selector: str
    contrast: float
    signal_frac: float
    seeds: int
    fp_mean: float
    fp_std: float
    tp_mean: float
    nan_runs: int


def _blank_sigma(rng_seed: int, noise_key: str, sigma: float) -> float:
    """Noise scale estimated from an independent blank, as Currie requires.

    The point of a critical level is that it is derived from a separate
    measurement of the signal-free case, not from the image under test.
    """
    rng = np.random.default_rng(rng_seed + 99991)
    blank = NOISE[noise_key](rng, SIZE, sigma)
    return float(blank.mean()), float(blank.std())


def run_signal_free(sigma: float = 0.03) -> list[Row]:
    """Experiment 1: the correct answer is zero everywhere."""
    rows: list[Row] = []
    for noise_key in NOISE:
        for sel_name, sel in SELECTORS.items():
            fps, nans = [], 0
            for s in range(SEEDS):
                rng = np.random.default_rng(1000 + s)
                img = NOISE[noise_key](rng, SIZE, sigma)
                t = sel(img)
                if not np.isfinite(t):
                    nans += 1
                    continue
                fps.append(float((img > t).mean()))
            rows.append(
                Row(
                    experiment="signal_free",
                    noise=noise_key,
                    selector=sel_name,
                    contrast=0.0,
                    signal_frac=0.0,
                    seeds=SEEDS,
                    fp_mean=float(np.mean(fps)) if fps else float("nan"),
                    fp_std=float(np.std(fps)) if fps else float("nan"),
                    tp_mean=float("nan"),
                    nan_runs=nans,
                )
            )

        # the absolute-floor reference, on the same inputs
        fps = []
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            img = NOISE[noise_key](rng, SIZE, sigma)
            mu, sd = _blank_sigma(1000 + s, noise_key, sigma)
            t = mu + CURRIE_K * sd
            fps.append(float((img > t).mean()))
        rows.append(
            Row(
                experiment="signal_free",
                noise=noise_key,
                selector=f"currie_{CURRIE_K:g}sigma",
                contrast=0.0,
                signal_frac=0.0,
                seeds=SEEDS,
                fp_mean=float(np.mean(fps)),
                fp_std=float(np.std(fps)),
                tp_mean=float("nan"),
                nan_runs=0,
            )
        )
    return rows


def _with_signal(
    rng: np.random.Generator, noise_key: str, sigma: float, contrast: float, frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """A square patch of raised magnitude, covering `frac` of the image."""
    img = NOISE[noise_key](rng, SIZE, sigma)
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    if frac > 0 and contrast > 0:
        side = int(round(np.sqrt(frac) * SIZE))
        if side > 0:
            o = (SIZE - side) // 2
            mask[o : o + side, o : o + side] = True
            img = img.copy()
            img[mask] += contrast * sigma
    return img, mask


def run_contrast_sweep(
    sigma: float = 0.03, frac: float = 0.01, contrasts=None
) -> list[Row]:
    """Experiment 2: raise the signal from nothing and watch the null appear.

    False positives are counted only on true-background pixels, so a selector
    is never penalised for finding the signal.
    """
    if contrasts is None:
        contrasts = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    rows: list[Row] = []
    for noise_key in ("rayleigh", "gaussian"):
        for sel_name, sel in SELECTORS.items():
            for c in contrasts:
                fps, tps, nans = [], [], 0
                for s in range(SEEDS):
                    rng = np.random.default_rng(2000 + s)
                    img, mask = _with_signal(rng, noise_key, sigma, c, frac)
                    t = sel(img)
                    if not np.isfinite(t):
                        nans += 1
                        continue
                    hit = img > t
                    bg = ~mask
                    fps.append(float(hit[bg].mean()))
                    tps.append(float(hit[mask].mean()) if mask.any() else float("nan"))
                rows.append(
                    Row(
                        experiment="contrast_sweep",
                        noise=noise_key,
                        selector=sel_name,
                        contrast=float(c),
                        signal_frac=frac,
                        seeds=SEEDS,
                        fp_mean=float(np.mean(fps)) if fps else float("nan"),
                        fp_std=float(np.std(fps)) if fps else float("nan"),
                        tp_mean=(
                            float(np.nanmean(tps))
                            if any(t == t for t in tps)
                            else float("nan")
                        ),
                        nan_runs=nans,
                    )
                )
    return rows


def run_floor(sigma: float = 0.03, frac: float = 0.01) -> list[Row]:
    """Experiment 3: max(selector, critical level), same inputs as above."""
    rows: list[Row] = []
    noise_key = "rayleigh"
    for sel_name, sel in SELECTORS.items():
        for c in (0.0, 1.0, 3.0, 8.0, 20.0):
            fps, tps = [], []
            for s in range(SEEDS):
                rng = np.random.default_rng(2000 + s)
                img, mask = _with_signal(rng, noise_key, sigma, c, frac)
                t = sel(img)
                mu, sd = _blank_sigma(2000 + s, noise_key, sigma)
                floor = mu + CURRIE_K * sd
                t = floor if not np.isfinite(t) else max(t, floor)
                hit = img > t
                bg = ~mask
                fps.append(float(hit[bg].mean()))
                tps.append(float(hit[mask].mean()) if mask.any() else float("nan"))
            rows.append(
                Row(
                    experiment="with_floor",
                    noise=noise_key,
                    selector=sel_name,
                    contrast=float(c),
                    signal_frac=frac,
                    seeds=SEEDS,
                    fp_mean=float(np.mean(fps)),
                    fp_std=float(np.std(fps)),
                    tp_mean=(
                        float(np.nanmean(tps))
                        if any(t == t for t in tps)
                        else float("nan")
                    ),
                    nan_runs=0,
                )
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/results")
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--frac", type=float, default=0.01)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    rows += run_signal_free(args.sigma)
    rows += run_contrast_sweep(args.sigma, args.frac)
    rows += run_floor(args.sigma, args.frac)

    import skimage

    payload = {
        "meta": {
            "size": SIZE,
            "seeds": SEEDS,
            "nbins": NBINS,
            "sigma": args.sigma,
            "signal_frac": args.frac,
            "currie_k": CURRIE_K,
            "numpy": np.__version__,
            "scikit_image": skimage.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "rows": [asdict(r) for r in rows],
    }
    (out / "null_sweep.json").write_text(json.dumps(payload, indent=2))

    # headline table: signal-free, Rayleigh
    print(f"\nSignal-free input. Correct answer: 0.00% of the image.\n")
    print(f"{'selector':<20} {'reported changed':>18}")
    print("-" * 40)
    for r in rows:
        if r.experiment == "signal_free" and r.noise == "rayleigh":
            print(f"{r.selector:<20} {r.fp_mean * 100:>15.2f}% ±{r.fp_std * 100:.2f}")

    print(f"\nWrote {out / 'null_sweep.json'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
