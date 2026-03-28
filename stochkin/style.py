"""Publication-quality Matplotlib style for stochkin.

The defaults reproduce the look-and-feel of the FES_2D.ipynb analysis
notebook (Arial font, inward ticks, 300 dpi, single-column figure width,
…).  Two entry-points are provided:

* :func:`set_publication_style` — apply the style globally.
* :func:`publication_style` — context manager that restores the
  previous rcParams on exit.

Quick start::

    import stochkin as sk
    sk.set_publication_style()          # global
    # or
    with sk.publication_style():        # scoped
        fig, ax = plt.subplots()
        ...
"""
from __future__ import annotations

import contextlib
import os
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── Default publication parameters ─────────────────────────────────────
# These mirror the final (overriding) rcParams block in FES_2D.ipynb.

_PUBLICATION_RC: dict = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "font.style": "normal",
    "font.variant": "normal",
    "font.weight": 400,
    # Figure
    "figure.dpi": 300,
    "figure.figsize": (3.3, 2.8),
    "figure.titlesize": 10,
    "figure.autolayout": True,
    # Axes
    "axes.facecolor": "white",
    "axes.linewidth": 0.75,
    "axes.titlesize": 6,
    "axes.labelsize": 7,
    "axes.labelpad": 0.5,
    "axes.grid": False,
    "axes.axisbelow": "line",
    # Lines
    "lines.linewidth": 1.0,
    # X ticks
    "xtick.direction": "in",
    "xtick.major.size": 5,
    "xtick.minor.size": 3,
    "xtick.major.width": 0.8,
    "xtick.minor.width": 0.8,
    "xtick.major.pad": 4,
    "xtick.labelsize": 8,
    # Y ticks
    "ytick.right": True,
    "ytick.direction": "in",
    "ytick.major.size": 5,
    "ytick.minor.size": 3,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.8,
    "ytick.major.pad": 4,
    "ytick.labelsize": 8,
    # Legend
    "legend.fontsize": 6,
    # Saving
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def _try_register_arial() -> None:
    """Try to register the system Arial font with Matplotlib (Linux)."""
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)


def set_publication_style(rc_overrides: Optional[dict] = None) -> None:
    """Apply the stochkin publication rcParams globally.

    Parameters
    ----------
    rc_overrides : dict, optional
        Extra key/value pairs that override the defaults.
    """
    _try_register_arial()
    rc = dict(_PUBLICATION_RC)
    if rc_overrides:
        rc.update(rc_overrides)
    mpl.rcParams.update(rc)


@contextlib.contextmanager
def publication_style(rc_overrides: Optional[dict] = None):
    """Context manager that temporarily applies the publication style.

    On exit, the previous rcParams are restored::

        with publication_style():
            fig, ax = plt.subplots()
            ...
    """
    old = mpl.rcParams.copy()
    try:
        set_publication_style(rc_overrides)
        yield
    finally:
        mpl.rcParams.update(old)


# ── Convenience tick/label size constants (for inline overrides) ──────
# These match the values that the notebook cells apply with explicit
# ``ax.tick_params(...)`` and ``ax.set_xlabel(..., size=...)``.
LABEL_SIZE = 12       # ax.set_xlabel(…, size=LABEL_SIZE)
TICK_SIZE = 10        # ax.tick_params(axis="both", labelsize=TICK_SIZE)
LEGEND_SIZE = 8       # ax.legend(fontsize=LEGEND_SIZE)
CBAR_LABEL_SIZE = 14
CBAR_TICK_SIZE = 10
TITLE_SIZE = 10


__all__ = [
    "set_publication_style",
    "publication_style",
    "_PUBLICATION_RC",
    "LABEL_SIZE",
    "TICK_SIZE",
    "LEGEND_SIZE",
    "CBAR_LABEL_SIZE",
    "CBAR_TICK_SIZE",
    "TITLE_SIZE",
]
