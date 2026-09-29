"""Figures for the null-test paper.

Three figures, each doing one job:

  fig1  magnitude across categories -> horizontal bars, sorted, direct labels
  fig2  change along a continuous variable, 5 series -> lines, direct labels
  fig3  two conditions across sites -> paired dot plot, one row per site

Palette is the site's own house hues, re-stepped until the validator passed all
six checks against a white surface (lightness band, chroma floor, CVD
separation, normal-vision floor, contrast). Text never wears a series colour;
identity is carried by the mark and repeated in a direct label, so nothing here
depends on colour alone.

Run:  .venv-paper/bin/python paper/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "figures"

# Validated categorical palette, in assignment order. Do not cycle it.
BLUE, RED, PURPLE, AMBER, GREEN = "#2a78d6", "#e34948", "#4a3aa7", "#a06f00", "#0f8f63"

INK = "#111111"
INK_MUTED = "#6b6b6b"
INK_FAINT = "#b8b8b8"
SURFACE = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.edgecolor": INK_FAINT,
    "axes.linewidth": 0.8,
})


def _strip(ax, keep_left=False):
    """Recessive axes: no box, no ticks, grid only where it earns its place."""
    for side in ("top", "right", "bottom" if not keep_left else "top"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.tick_params(length=0)


def fig1_signal_free() -> None:
    """What each selector returns when the correct answer is zero."""
    rows = json.loads((RESULTS / "null_sweep.json").read_text())["rows"]
    r = [x for x in rows if x["experiment"] == "signal_free" and x["noise"] == "rayleigh"]
    r.sort(key=lambda x: x["fp_mean"], reverse=True)

    LABELS = {
        "li": "Li", "mean": "mean", "ridler_calvard": "Ridler-Calvard", "otsu": "Otsu",
        "yen": "Yen", "kapur": "Kapur", "triangle_rosin": "Rosin triangle",
        "currie_3sigma": "Currie 3σ floor",
    }
    names = [LABELS.get(x["selector"], x["selector"]) for x in r]
    vals = [x["fp_mean"] * 100 for x in r]
    # the floor is not a selector; it is the contrast the figure exists to draw
    colors = [RED if not x["selector"].startswith("currie") else GREEN for x in r]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = range(len(r))
    ax.barh(list(y), vals, height=0.62, color=colors, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.9, i, f"{v:.2f}%", va="center", ha="left",
                fontsize=8.5, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("percentage of a signal-free scene reported as changed")
    ax.axvline(0, color=INK_FAINT, lw=0.8, zorder=2)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50%"])
    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
    _strip(ax)
    ax.set_title("The correct answer is 0.00%", loc="left", fontsize=10,
                 color=INK, pad=10, fontweight="medium")
    fig.tight_layout()
    fig.savefig(OUT / "null-test-fig1.png", dpi=200)
    plt.close(fig)
    print("fig1 written")


def fig2_regime() -> None:
    """False positives as a real signal is raised from nothing.

    Plotted from a dense sweep, 0.25 steps, because the original coarse grid
    drew a straight line between contrast 5 and 8 and that line was
    interpolation rather than measurement. Resolved properly, Otsu's fall is a
    genuine discontinuity and Li's is not, which is a claim the coarse figure
    could not have supported either way.
    """
    d = json.loads((RESULTS / "dense_sweep.json").read_text())
    xs = d["contrasts"]
    series = [
        ("li", "Li", PURPLE), ("ridler_calvard", "Ridler-Calvard", AMBER),
        ("otsu", "Otsu", RED), ("kapur", "Kapur", BLUE),
        ("triangle_rosin", "Rosin triangle", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))

    labels = []
    for key, label, color in series:
        ys = [v * 100 for v in d["series"][key]]
        ax.plot(xs, ys, lw=2.0, color=color, zorder=3)
        labels.append([ys[0], label, color])

    # de-collide the left-edge labels: nothing may sit within 2.6 points of
    # its neighbour, so Ridler-Calvard and Otsu stop overprinting each other
    labels.sort(key=lambda r: r[0])
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < 2.6:
            labels[i][0] = labels[i - 1][0] + 2.6
    for y, label, color in labels:
        ax.text(-0.5, y, label, ha="right", va="center", fontsize=8.5, color=INK)

    # name the discontinuity rather than leaving the reader to find it
    ax.annotate(
        "26.97 points\nin one 0.25σ step",
        xy=(6.18, 14.0), xytext=(2.4, 15.5),
        fontsize=8, color=INK_MUTED, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=INK_FAINT, lw=0.9,
                        connectionstyle="arc3,rad=0.0"),
    )

    ax.set_xlim(-4.3, 14.3)
    ax.set_ylim(-2, 55)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_xlabel("contrast of the injected signal (multiples of the noise scale)")
    ax.set_ylabel("false positives on true-background pixels")
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_yticklabels(["0", "10", "20", "30", "40", "50%"])
    ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
    _strip(ax)
    ax.set_title("Otsu does not decay. It falls off a cliff.", loc="left",
                 fontsize=10, color=INK, pad=10, fontweight="medium")
    fig.tight_layout()
    fig.savefig(OUT / "null-test-fig2.png", dpi=200)
    plt.close(fig)
    print("fig2 written")


def fig3_field() -> None:
    """Real Sentinel-2 over the six CEOS-endorsed invariant sites."""
    rows = [r for r in json.loads((RESULTS / "field_null.json").read_text())
            if r["status"] == "ok" and r["certified_pics"]]
    sites = []
    for r in rows:
        if r["site"] not in sites:
            sites.append(r["site"])

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, site in enumerate(sites):
        rs = [r for r in rows if r["site"] == site]
        ax.plot([0, 55], [i, i], color="#f2f2f2", lw=0.8, zorder=0)
        for r in rs:
            ax.plot(r["no_floor_pct"], i, "o", markersize=7, color=RED,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
            ax.plot(r["calibrated_pct"], i, "o", markersize=7, color=BLUE,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)

    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels([s.replace("-", " ").title() for s in sites],
                       fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-1.5, 55)
    ax.set_xlabel("percentage of the scene reported as changed")
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50%"])
    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
    _strip(ax)

    handles = [
        plt.Line2D([], [], marker="o", ls="", markersize=7, color=BLUE,
                   markeredgecolor=SURFACE, markeredgewidth=1.5,
                   label="with an absolute floor"),
        plt.Line2D([], [], marker="o", ls="", markersize=7, color=RED,
                   markeredgecolor=SURFACE, markeredgewidth=1.5,
                   label="threshold chosen from the data"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False,
                    fontsize=8.5, handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.set_title("Ground certified stable to 0.215% per year", loc="left",
                 fontsize=10, color=INK, pad=10, fontweight="medium")
    fig.tight_layout()
    fig.savefig(OUT / "null-test-fig3.png", dpi=200)
    plt.close(fig)
    print("fig3 written")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_signal_free()
    fig2_regime()
    fig3_field()
    print(f"-> {OUT}")
