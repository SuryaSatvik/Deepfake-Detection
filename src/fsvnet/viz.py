"""
viz.py
======

One place where every figure in this project gets its colour, type and spacing,
so the notebook, the evaluation figures and the evidence packs read as one
system instead of eight different matplotlib defaults.

Rules this file enforces (they are not stylistic preferences — they are the
difference between a figure a reviewer can read and one they can't):

* **Categorical hues are assigned in a fixed order, never cycled.** Slot 1 is
  always the first series in every chart, so "blue" means the same thing on
  every page. The order below is CVD-validated: adjacent pairs stay separable
  for red-green and blue-yellow colour vision deficiency.
* **Sequential encoding is one hue, light → dark.** Heatmaps of a magnitude
  (a TSC map, a confusion matrix) use the blue ramp — never a rainbow and
  never `jet`/`magma`, which invent perceptual edges that are not in the data.
* **Diverging encoding is two hues around a neutral grey**, used only where
  zero is genuinely meaningful.
* **Text never wears the series colour.** Values and labels stay in ink; the
  coloured mark beside them carries the identity.
* **Grid and axes are recessive.** The data is the loudest thing on the page.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

# Categorical slots, in fixed assignment order.
SERIES: List[str] = [
    "#2a78d6",   # 1 blue
    "#eb6834",   # 2 orange
    "#1baf7a",   # 3 aqua
    "#eda100",   # 4 yellow
    "#e87ba4",   # 5 magenta
    "#008300",   # 6 green
    "#4a3aa7",   # 7 violet
    "#e34948",   # 8 red
]

# Fixed meaning across every figure in the project, so a colour never changes
# what it refers to between two charts on the same page.
BRANCH_COLOR: Dict[str, str] = {
    "spatial": SERIES[0],
    "frequency": SERIES[1],
    "srm": SERIES[2],
    "tsc": SERIES[6],          # violet: the novel branch, visually distinct
}
CLASS_COLOR: Dict[str, str] = {
    "real": SERIES[0],
    "fake": SERIES[1],
    "smooth": SERIES[1],
    "flicker": SERIES[2],
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e0"

# Sequential ramp: one hue, light -> dark.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
             "#184f95", "#0d366b"]
# Diverging: two hues + neutral grey midpoint. Zero must be meaningful to use it.
DIVERGING = ["#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948",
             "#8f2020"]

STATUS = {"good": "#008300", "warning": "#eda100",
          "serious": "#eb6834", "critical": "#e34948"}


def sequential_cmap(name: str = "fsv_blue"):
    """The project's one sequential colormap. Use this, not `viridis`/`magma`."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, BLUE_RAMP)


def diverging_cmap(name: str = "fsv_div"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, DIVERGING)


# --------------------------------------------------------------------------
# Global style
# --------------------------------------------------------------------------

def use_style() -> None:
    """Call once at the top of a notebook or script."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "font.family": "DejaVu Sans",
    })



def _ensure_backend() -> None:
    """
    Pick a rendering backend exactly once, and never clobber an interactive one.

    This function exists because of a real bug: the figure helpers used to call
    `matplotlib.use("Agg")` on every invocation. In a script that is harmless;
    in a notebook it silently switches the backend mid-session, and every
    `plt.show()` in every later cell renders nothing at all — no error, no
    figure, just a missing plot. So: only fall back to Agg when there is no
    display AND we are not inside IPython.
    """
    import matplotlib

    current = matplotlib.get_backend().lower()
    if "inline" in current or "nbagg" in current or "widget" in current:
        return                       # a notebook already chose; leave it alone
    try:
        get_ipython                  # noqa: F821  — defined only inside IPython
        return
    except NameError:
        pass
    if current.startswith("agg"):
        return
    import os
    if os.environ.get("DISPLAY"):
        return
    matplotlib.use("Agg")


def despine(ax, keep: Sequence[str] = ("left", "bottom")) -> None:
    """Drop the box; keep only the spines that carry information."""
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(side in keep)


def bar_labels(ax, bars, fmt: str = "{:.3f}", horizontal: bool = False,
               pad: float = 0.008, color: Optional[str] = None) -> None:
    """
    Direct-label bars in ink, not in the series colour — the mark carries the
    identity, the text carries the value.
    """
    color = color or INK
    for b in bars:
        if horizontal:
            w = b.get_width()
            ax.text(w + pad, b.get_y() + b.get_height() / 2, fmt.format(w),
                    va="center", ha="left", fontsize=9, color=color)
        else:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h),
                    ha="center", va="bottom", fontsize=9, color=color)


def chance_line(ax, y: float = 0.5, horizontal: bool = True,
                label: str = "chance") -> None:
    """A reference line for AUC/accuracy charts. Recessive, always labelled."""
    fn = ax.axhline if horizontal else ax.axvline
    fn(y, ls=(0, (4, 3)), lw=1.2, color=INK_MUTED, zorder=1)
    if horizontal:
        ax.text(ax.get_xlim()[1], y, f" {label}", va="center", ha="left",
                fontsize=8.5, color=INK_MUTED)
    else:
        ax.text(y, ax.get_ylim()[1], f" {label}", va="bottom", ha="center",
                fontsize=8.5, color=INK_MUTED)


def stat_tiles(specs: List[dict], ncols: int = 4, figsize=None):
    """
    A row of headline numbers. Use when the answer is one value per box and a
    chart would only decorate it.

    specs: [{"value": "0.573", "label": "test AUC", "note": "video level",
             "status": "good"|"warning"|"critical"|None}, ...]
    """
    import matplotlib.pyplot as plt

    n = len(specs)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    figsize = figsize or (3.1 * ncols, 1.65 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = [axes] if n == 1 else list(axes.ravel())

    for ax, s in zip(axes, specs):
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        col = STATUS.get(s.get("status") or "", INK)
        ax.text(0.02, 0.62, str(s["value"]), fontsize=25, fontweight="600",
                color=col, va="center", ha="left", transform=ax.transAxes)
        ax.text(0.02, 0.25, s["label"], fontsize=9.5, color=INK_2,
                va="center", ha="left", transform=ax.transAxes)
        if s.get("note"):
            ax.text(0.02, 0.06, s["note"], fontsize=8.5, color=INK_MUTED,
                    va="center", ha="left", transform=ax.transAxes)
        ax.plot([0.02, 0.98], [0.94, 0.94], lw=2.5, color=col,
                transform=ax.transAxes, clip_on=False, solid_capstyle="butt")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


__all__ = ["_ensure_backend", "SERIES", "BRANCH_COLOR", "CLASS_COLOR", "SURFACE", "INK", "INK_2",
           "INK_MUTED", "GRID", "BLUE_RAMP", "DIVERGING", "STATUS",
           "sequential_cmap", "diverging_cmap", "use_style", "despine",
           "bar_labels", "chance_line", "stat_tiles"]
