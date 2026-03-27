"""Stochastic_Estimation.boundaries

Small, picklable boundary-condition helpers.

Why this module exists
----------------------
Several workflows in this project (cropped FES domains, ROI sampling, MFPT
estimation) need a *consistent* way to keep trajectories inside a rectangular
box.

The mirror-reflection rule is the same as in your pasted script:
if a coordinate exits [lo, hi] it is reflected back in (possibly multiple times
if the step overshoots by more than the box size).

Notes
-----
- For overdamped dynamics, reflection acts only on positions.
- For underdamped dynamics, if you reflect a coordinate you should also flip
  the corresponding velocity component (elastic reflection). This is provided
  by :func:`reflect_position_velocity`.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


Bounds = Sequence[Tuple[float, float]]


def _as_1d_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Empty position.")
    return arr


def reflect_scalar(x: float, lo: float, hi: float) -> Tuple[float, int]:
    """Mirror-reflect a scalar into [lo, hi].

    Returns
    -------
    x_reflected : float
    n_flips : int
        Number of reflections performed (parity matters for velocity flips).
    """
    lo = float(lo)
    hi = float(hi)
    if lo >= hi:
        raise ValueError("Invalid interval [lo, hi].")

    x = float(x)
    n_flips = 0

    # robust for large excursions; parity of flips is what matters for velocities
    while x < lo or x > hi:
        if x < lo:
            x = 2.0 * lo - x
            n_flips += 1
        if x > hi:
            x = 2.0 * hi - x
            n_flips += 1

    return x, n_flips


def apply_bounds(x, bounds: Optional[Bounds], mode: str = "reflect") -> np.ndarray:
    """Enforce rectangular bounds on a position vector.

    Parameters
    ----------
    x : array_like, shape (d,)
    bounds : sequence of (lo, hi), length d
        Example for 2D: ((xlo, xhi), (ylo, yhi)).
        If None, returns x unchanged.
    mode : {'reflect', 'clip'}

    Returns
    -------
    x_new : ndarray, shape (d,)
    """
    x = _as_1d_array(x).copy()

    if bounds is None:
        return x

    if len(bounds) != x.size:
        raise ValueError("bounds must have one (lo,hi) pair per coordinate.")

    if mode not in ("reflect", "clip"):
        raise ValueError("mode must be 'reflect' or 'clip'.")

    for i, (lo, hi) in enumerate(bounds):
        lo = float(lo)
        hi = float(hi)
        if lo >= hi:
            raise ValueError("Invalid bounds interval: lo must be < hi.")

        if mode == "reflect":
            x[i], _ = reflect_scalar(x[i], lo, hi)
        else:
            x[i] = float(np.clip(x[i], lo, hi))

    return x


def reflect_position_velocity(
    x,
    v,
    bounds: Optional[Bounds],
) -> Tuple[np.ndarray, np.ndarray]:
    """Reflect position into bounds and flip velocity components accordingly.

    This is a simple *elastic wall* model compatible with mirror reflection.
    It is useful for keeping underdamped trajectories inside a cropped box.

    Parameters
    ----------
    x, v : array_like, shape (d,)
    bounds : sequence of (lo, hi), length d

    Returns
    -------
    x_new, v_new : ndarrays
    """
    x = _as_1d_array(x).copy()
    v = _as_1d_array(v).copy()

    if bounds is None:
        return x, v

    if len(bounds) != x.size:
        raise ValueError("bounds must have one (lo,hi) pair per coordinate.")

    for i, (lo, hi) in enumerate(bounds):
        x_i, n_flips = reflect_scalar(x[i], lo, hi)
        x[i] = x_i
        if n_flips % 2 == 1:
            v[i] = -v[i]

    return x, v
