# What does a change detector find where nothing changed?

Code, results and figures for the paper
**[What Does a Change Detector Find Where Nothing Changed?](https://www.subomi.net/papers/what-a-change-detector-finds)**
(Subomi Olagoke, 2026).

The paper proposes a null test for satellite change detection: run the detector
over ground that the Committee on Earth Observation Satellites (CEOS) already
certifies as radiometrically invariant, so that every detection is a false alarm
by construction. It pairs that with an injection test, painting a structure of
known size into real imagery, to get a detection floor. Neither needs a single
hand label.

## Contents

| Path | What it is |
|---|---|
| `paper/null_sweep.py` | Experiments 1–3: what threshold selectors return on signal-free input, as a signal appears, and with a Currie critical-level floor. Synthetic; no downloads. |
| `paper/field_null.py` | Experiments 4–5: the pipeline over the six CEOS-endorsed pseudo-invariant sites, across temporal separations. Downloads free Sentinel-2 imagery. |
| `paper/injection.py` | Experiment 6: the injection test on the same pairs. Run `field_null.py` first. |
| `paper/figures.py` | Figures 1–3 from the results files. |
| `paper/results/` | The result files the paper reports. |
| `paper/figures/` | The figures as published. |
| `juwon/` | The subset of the jùwọ̀n change-detection engine these experiments call: change detection, geometry, and the Sentinel-1/2 imagery providers. The rest of jùwọ̀n (API, app) is not needed and not included. |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python paper/null_sweep.py      # ~30 s, reproduces results/null_sweep.json exactly
python paper/field_null.py      # downloads Sentinel-2 scenes; slow on first run, cached after
python paper/injection.py
python paper/figures.py
```

Run the scripts from the repository root. Imagery comes from the public
[Element 84 Earth Search](https://earth-search.aws.element84.com/v1) STAC catalogue
(Sentinel-2) and Microsoft Planetary Computer (Sentinel-1, anonymous tokens). No
API keys, accounts or GPU are needed. Downloads are cached in `.data/`.

## Scope

The detector here is a classical brightness-differencing baseline, not a learned
model. The paper's claims are about the test and about this pipeline; running the
same null sites against learned change detectors is the obvious next step and has
not been done here.

## Licences

- Code (`juwon/`, `paper/*.py`): MIT, see `LICENSE`.
- Results and figures (`paper/results/`, `paper/figures/`): CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/).
- Results derived from satellite imagery contain modified Copernicus Sentinel data
  (2024), processed by the author.

## Citing

Olagoke, S. (2026). *What Does a Change Detector Find Where Nothing Changed?*
https://www.subomi.net/papers/what-a-change-detector-finds
