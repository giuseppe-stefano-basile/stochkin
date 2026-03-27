# potentials.py

import numpy as np
from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from functools import partial   # <--- ADD THIS
import warnings


# =========================
# Analytic potentials
# =========================
class StringPotential:
    """
    Picklable potential built from a string expression, e.g.:

        expr = "0.5*(x[0]**2 + 2*x[1]**2)"

    Use as:
        pot = StringPotential(expr)
        U, F = pot(x)

    where x can be 1D or 2D (or higher) NumPy-like array.
    """

    def __init__(self, expr, eps=1e-6):
        self.expr = expr
        self.eps = eps

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        energy = eval(self.expr, {"x": x, "np": np})

        eps = self.eps
        force = np.zeros_like(x)
        for i in range(len(x)):
            dx = np.zeros_like(x)
            dx[i] = eps
            e_plus = eval(self.expr, {"x": x + dx, "np": np})
            e_minus = eval(self.expr, {"x": x - dx, "np": np})
            force[i] = -(e_plus - e_minus) / (2 * eps)

        return energy, force


def make_potential_from_string(expr, eps=1e-6):
    """
    expr: string, e.g. "0.5*(x[0]**2 + 2*x[1]**2)"
    Returns a *picklable* object that computes (energy, force).
    """
    return StringPotential(expr, eps=eps)


def double_well_2d(x, a=1.0, b=1.3):
    x = np.asarray(x, dtype=float)
    U = a * (x[0] ** 4 + x[1] ** 4) - b * (x[0] ** 2 + x[1] ** 2)
    F = -np.array(
        [
            4 * a * x[0] ** 3 - 2 * b * x[0],
            4 * a * x[1] ** 3 - 2 * b * x[1],
        ]
    )
    return U, F


def mexican_hat_potential(x, a=-1.0, b=1.0):
    """
    2D Mexican hat potential.
    x : np.array([x1, x2])
    a : quadratic coefficient (negative for sombrero shape)
    b : quartic coefficient (positive for stability)
    Returns: (energy, force)
    """
    x = np.asarray(x, dtype=float)
    r2 = np.dot(x, x)  # |x|^2
    energy = a * r2 + b * r2 ** 2
    # Force = -∇U = -(2a x + 4b r^2 x)
    force = -(2 * a * x + 4 * b * r2 * x)
    return energy, force


def central_well_barrier_ring_potential(
    x, a=1.0, b=1.0, A=0.25, sigma=0.20
):
    """
    2D potential with a center well, a barrier, and ring minima.
    U(r) = b*r^4 - a*r^2 - A*exp(-r^2/sigma^2),  r = |x|

    Parameters:
      a > 0, b > 0  : set ring structure (sombrero-like)
      A > 0         : depth of central well (Gaussian bump down)
      sigma > 0     : width of central well

    Returns:
      (energy, force) where force is a 2D vector
    """
    x = np.asarray(x, dtype=float)
    r2 = np.dot(x, x)
    energy = b * r2 ** 2 - a * r2 - A * np.exp(-r2 / (sigma ** 2))
    # dU/dr = 4*b*r^3 - 2*a*r + (2*A/sigma^2)*r*exp(-r^2/sigma^2)
    # Force vector: F = -∇U = -(dU/dr) * x / r =
    #  -(4*b*r^2 - 2*a + (2*A/sigma**2)*exp(-r2/sigma**2)) * x
    coeff = -(4 * b * r2 - 2 * a + (2 * A / (sigma ** 2)) * np.exp(-r2 / (sigma ** 2)))
    force = coeff * x
    return energy, force


def muller_potential(x, E_clip=50.0):
    """
    Müller–Brown potential with analytic force and numerical safeguards.
    x : array-like [x, y]
    Returns: (energy, force)
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (2,):
        raise ValueError("Position must be a 2D vector [x, y].")

    # If x is already NaN/inf, just return NaNs to make the problem obvious
    if not np.all(np.isfinite(x)):
        return np.nan, np.array([np.nan, np.nan])

    X, Y = x

    # Standard MB parameters
    A = np.array([-200.0, -100.0, -170.0, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x0 = np.array([1.0, 0.0, -0.5, -1.0])
    y0 = np.array([0.0, 0.5, 1.5, 1.0])

    dx = X - x0
    dy = Y - y0

    # Exponent arguments for each Gaussian
    E = a * dx ** 2 + b * dx * dy + c * dy ** 2

    # Clip E to avoid overflow in exp (double precision overflows ~709)
    E_clipped = np.clip(E, -E_clip, E_clip)

    eE = np.exp(E_clipped)

    # Energy
    energy = np.sum(A * eE)

    # Derivatives of E
    dEdx = 2.0 * a * dx + b * dy
    dEdy = b * dx + 2.0 * c * dy

    # Gradient of U
    dUdx = np.sum(A * eE * dEdx)
    dUdy = np.sum(A * eE * dEdy)

    # Force: F = -∇U
    force = -np.array([dUdx, dUdy])

    return energy, force


def simple_double_well_2d(x, a=10.0, x0=1.0, k_y=10):
    """
    Simple 2-state double well in 2D.

    U(x, y) = a (x^2 - x0^2)^2 + 0.5 * k_y * y^2

    - Two minima at (x, y) = (±x0, 0)
    - Barrier at (0, 0) of height a * x0^4
    - Harmonic confinement along y

    Parameters
    ----------
    x : array-like, shape (2,)
        [x, y] coordinates
    a : float
        Double well strength in x
    x0 : float
        Location of the minima along x
    k_y : float
        Harmonic stiffness along y

    Returns
    -------
    U : float
        Potential energy
    F : ndarray, shape (2,)
        Force vector = -∇U
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (2,):
        raise ValueError("Position must be a 2D vector [x, y].")

    x1, x2 = x

    # Potential
    U = a * (x1**2 - x0**2)**2 + 0.5 * k_y * x2**2

    # Derivatives:
    # dU/dx1 = 4 a x1 (x1^2 - x0^2)
    # dU/dx2 = k_y x2
    dUdx1 = 4.0 * a * x1 * (x1**2 - x0**2)
    dUdx2 = k_y * x2

    F = -np.array([dUdx1, dUdx2])
    return U, F


def double_well_1d(x, a=1.0, x0=1.0):
    """
    Simple symmetric 1D double-well potential.

    U(x) = a (x^2 - x0^2)^2

    - Two minima at x = ±x0
    - Barrier at x = 0 of height a * x0^4

    Parameters
    ----------
    x : scalar or array-like, shape (1,)
        Position along the 1D coordinate.
    a : float
        Double-well strength.
    x0 : float
        Position of the minima.

    Returns
    -------
    U : float
        Potential energy at x.
    F : ndarray, shape (1,)
        Force = -dU/dx as a 1D vector.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.size != 1:
        raise ValueError("double_well_1d expects a scalar or length-1 array for x.")

    x_val = float(x_arr[0])

    # Potential
    U = a * (x_val**2 - x0**2) ** 2

    # dU/dx = 4 a x (x^2 - x0^2)
    dUdx = 4.0 * a * x_val * (x_val**2 - x0**2)

    F = -np.array([dUdx], dtype=float)
    return U, F

def make_double_well_1d(a=1.0, x0=1.0):
    """
    Convenience wrapper that returns a picklable 1D double-well potential.

    We use functools.partial on the *top-level* function double_well_1d
    so that the result works with multiprocessing.Pool.
    """
    return partial(double_well_1d, a=a, x0=x0)



# =========================
# Basin / minima utilities
# =========================

@dataclass
class Basin:
    """
    Simple description of a basin around a local minimum.

    Attributes
    ----------
    id : int
        Basin index (0..N-1).
    minimum : np.ndarray shape (2,)
        Coordinates (x, y) of the minimum.
    f_min : float
        Potential/free-energy value at the minimum.
    radius : float
        Rough spatial extent of the basin (max distance of assigned grid points).
    bounds : np.ndarray shape (2,2)
        [[xmin, xmax], [ymin, ymax]] of assigned grid points.
    """
    id: int
    minimum: np.ndarray
    f_min: float
    radius: float
    bounds: np.ndarray


@dataclass
class BasinNetwork:
    """
    Basins defined on a regular 2D grid.

    Attributes
    ----------
    basins : list[Basin]
    xs, ys : 1D arrays of grid coordinates
    U      : 2D array of potential/free energy values (shape (nx, ny))
    labels : 2D int array, same shape as U.
             labels[i, j] is the basin id of grid point (xs[i], ys[j]),
             or -1 if the point does not belong to any basin (e.g. NaN).
    """
    basins: list
    xs: np.ndarray
    ys: np.ndarray
    U: np.ndarray
    labels: np.ndarray


    labels_full: Optional[np.ndarray] = None
    core_labels: Optional[np.ndarray] = None
    @property
    def n_basins(self) -> int:
        return len(self.basins)

    def which_basin(self, x):
        """
        Return basin id (int) for a position x = [x, y],
        or None if x is outside the grid or in an unlabeled region.
        """
        x = np.asarray(x, dtype=float)
        X, Y = float(x[0]), float(x[1])

        # Out of bounds -> no basin
        if X < self.xs[0] or X > self.xs[-1] or Y < self.ys[0] or Y > self.ys[-1]:
            return None

        # Find nearest grid index in each direction
        i = np.searchsorted(self.xs, X)
        if i == len(self.xs):
            i -= 1
        elif i > 0 and abs(X - self.xs[i - 1]) < abs(self.xs[i] - X):
            i -= 1

        j = np.searchsorted(self.ys, Y)
        if j == len(self.ys):
            j -= 1
        elif j > 0 and abs(Y - self.ys[j - 1]) < abs(self.ys[j] - Y):
            j -= 1

        label = int(self.labels[i, j])
        if label < 0:
            return None
        return label

    def sample_point_in_basin(self, basin_id, rng=None):
        """
        Sample a random (x, y) from grid points assigned to basin_id.
        Returns np.array([x, y]) or None if basin is empty.
        """
        if rng is None:
            rng = np.random

        mask = self.labels == basin_id
        idx = np.argwhere(mask)
        if idx.size == 0:
            return None

        k = rng.randint(idx.shape[0])
        i, j = idx[k]
        return np.array([self.xs[i], self.ys[j]])


@dataclass
class Basin1D:
    """
    1D basin around a local minimum.

    Attributes
    ----------
    id      : int
        Basin index (0..N-1).
    minimum : float
        Coordinate of the minimum.
    f_min   : float
        Potential / free-energy at the minimum.
    radius  : float
        Max |x - minimum| of assigned grid points.
    bounds  : np.ndarray shape (2,)
        [xmin, xmax] of assigned grid points.
    """
    id: int
    minimum: float
    f_min: float
    radius: float
    bounds: np.ndarray


@dataclass
class BasinNetwork1D:
    """
    Basins defined on a regular 1D grid.

    Attributes
    ----------
    basins : list[Basin1D]
    s      : 1D array of grid coordinates
    U      : 1D array of potential / FES values
    labels : 1D int array, same length as s
             labels[i] is the basin id of s[i],
             or -1 if the point does not belong to any basin.
    """
    basins: list
    s: np.ndarray
    U: np.ndarray
    labels: np.ndarray

    @property
    def n_basins(self) -> int:
        return len(self.basins)

    def which_basin(self, x):
        """
        Return basin id (int) for position x, or None if outside grid
        or in an unlabeled region.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            return None
        X = float(x[0])

        s = self.s
        # Out of bounds -> no basin
        if X < s[0] or X > s[-1]:
            return None

        # Nearest grid index
        i = np.searchsorted(s, X)
        if i == len(s):
            i -= 1
        # If X lies between s[i-1] and s[i], choose the nearer of the two.
        # NOTE: compare distances to the two grid points (not to the grid spacing).
        elif i > 0 and abs(X - s[i - 1]) < abs(s[i] - X):
            i -= 1

        label = int(self.labels[i])
        if label < 0:
            return None
        return label

    def sample_point_in_basin(self, basin_id, rng=None):
        """
        Sample a random x from grid points assigned to basin_id.
        Returns np.array([x]) or None if basin is empty.
        """
        if rng is None:
            rng = np.random

        mask = self.labels == basin_id
        idx = np.argwhere(mask)
        if idx.size == 0:
            return None

        k = rng.randint(idx.shape[0])
        i = idx[k, 0]
        return np.array([self.s[i]], dtype=float)

# ---------- low-level helpers ----------

def sample_potential_grid(
    potential,
    xlim=(-2.0, 2.0),
    ylim=(-2.0, 2.0),
    nx=200,
    ny=200,
):
    """Sample a 2D potential on a regular grid.

    For FES-based potentials this can be a hot path (basin detection).
    If the potential exposes a vectorized grid evaluator, use it.

    Parameters
    ----------
    potential : callable
        Potential callable that returns (U, F) for a single point.
    xlim, ylim : tuple
        Bounds of the grid.
    nx, ny : int
        Grid resolution.

    Returns
    -------
    xs, ys : 1D arrays
    U      : 2D array, shape (nx, ny)
    """
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)

    # Fast path: vectorized grid evaluation (e.g., FESPotential)
    eval_grid = getattr(potential, "evaluate_U_on_grid", None)
    if callable(eval_grid):
        U = eval_grid(xs, ys)
        return xs, ys, np.asarray(U, dtype=float)

    # Fallback: scalar calls
    U = np.empty((nx, ny), dtype=float)
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            u, _ = potential([xv, yv])
            U[i, j] = u

    return xs, ys, U


def _find_local_minima_grid(U, max_basins=None, sort_by_energy=True):
    """
    Find local minima on a 2D grid by strict comparison with 8 neighbors.

    U : 2D array
    Returns
    -------
    minima_idx : array of shape (n_minima, 2)
        Each row is (i, j) index of a local minimum.
    """
    U = np.asarray(U, dtype=float)
    # Treat NaNs/infs as very high so they are never minima
    U_eff = np.where(np.isfinite(U), U, np.inf)

    # Pad with +inf so borders can't be minima due to missing neighbors
    pad = np.pad(U_eff, 1, mode="constant", constant_values=np.inf)
    center = pad[1:-1, 1:-1]

    neighbors = [
        pad[:-2, 1:-1],   # up
        pad[2:, 1:-1],    # down
        pad[1:-1, :-2],   # left
        pad[1:-1, 2:],    # right
        pad[:-2, :-2],    # up-left
        pad[:-2, 2:],     # up-right
        pad[2:, :-2],     # down-left
        pad[2:, 2:],      # down-right
    ]

    mask = np.ones_like(center, dtype=bool)
    for N in neighbors:
        mask &= center < N

    minima_idx = np.argwhere(mask)  # indices in original U

    if minima_idx.size == 0:
        return minima_idx

    if sort_by_energy:
        energies = U_eff[minima_idx[:, 0], minima_idx[:, 1]]
        order = np.argsort(energies)
        minima_idx = minima_idx[order]

    if max_basins is not None and minima_idx.shape[0] > max_basins:
        minima_idx = minima_idx[:max_basins]

    return minima_idx


def _assign_labels_by_nearest_minima(xs, ys, minima_idx):
    """
    Label each grid point by nearest minimum (in Euclidean distance in CV space).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    nx, ny = len(xs), len(ys)

    X, Y = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)

    minima_idx = np.asarray(minima_idx, dtype=int)
    min_x = xs[minima_idx[:, 0]]
    min_y = ys[minima_idx[:, 1]]

    # Broadcast distances: shape (nx, ny, n_minima)
    dx = X[..., None] - min_x[None, None, :]
    dy = Y[..., None] - min_y[None, None, :]
    dist2 = dx ** 2 + dy ** 2

    labels = dist2.argmin(axis=2).astype(int)  # (nx, ny)
    return labels


def _find_local_minima_1d(F, max_basins=None, sort_by_energy=True):
    """
    Find local minima on a 1D grid by comparison with left/right neighbors.

    F : 1D array (potential or FES values)
    Returns
    -------
    minima_idx : 1D array of indices of local minima
    """
    F = np.asarray(F, dtype=float)
    # Treat NaNs/infs as very high so they are never minima
    F_eff = np.where(np.isfinite(F), F, np.inf)

    n = F_eff.size
    minima = []
    for i in range(n):
        left = F_eff[i - 1] if i > 0 else np.inf
        right = F_eff[i + 1] if i < n - 1 else np.inf
        if F_eff[i] < left and F_eff[i] < right:
            minima.append(i)

    minima = np.array(minima, dtype=int)
    if minima.size == 0:
        return minima

    if sort_by_energy:
        idx_sort = np.argsort(F_eff[minima])
        minima = minima[idx_sort]

    if max_basins is not None and minima.size > max_basins:
        minima = minima[:max_basins]

    # Sort by position along the grid
    minima = np.sort(minima)
    return minima


def _assign_labels_1d(n_points, minima_idx):
    """
    Assign each 1D grid point to the nearest minimum in index space.

    Returns labels[0..n_points-1] with basin ids or -1.
    """
    labels = -np.ones(n_points, dtype=int)
    if len(minima_idx) == 0:
        return labels

    mins = np.sort(np.asarray(minima_idx, dtype=int))
    nmin = len(mins)

    for k, m in enumerate(mins):
        if k == 0:
            start = 0
        else:
            start = (mins[k - 1] + m) // 2 + 1

        if k == nmin - 1:
            end = n_points - 1
        else:
            end = (m + mins[k + 1]) // 2

        labels[start : end + 1] = k

    return labels


def build_basin_network_from_fes_1d(
    s,
    F,
    max_basins=None,
    verbose=True,
):
    """
    Build a BasinNetwork1D directly from a 1D FES on a grid.

    Parameters
    ----------
    s : 1D array
        Grid points (e.g., CV values).
    F : 1D array
        Free energy (or potential) at each grid point.
    """
    s = np.asarray(s, dtype=float)
    F = np.asarray(F, dtype=float)
    if s.ndim != 1 or F.ndim != 1 or s.shape[0] != F.shape[0]:
        raise ValueError("s and F must be 1D arrays of the same length")

    minima_idx = _find_local_minima_1d(F, max_basins=max_basins, sort_by_energy=True)
    if minima_idx.size == 0:
        raise RuntimeError("No local minima found on the 1D FES; check grid / data.")

    labels = _assign_labels_1d(len(s), minima_idx)
    # Any NaN/inf regions get label -1
    labels[~np.isfinite(F)] = -1

    basins = []
    for basin_id, i_min in enumerate(minima_idx):
        mask = labels == basin_id
        if not np.any(mask):
            continue

        s_mask = s[mask]
        min_pos = float(s[i_min])
        f_min = float(F[i_min])
        radius = float(np.max(np.abs(s_mask - min_pos)))
        bounds = np.array(
            [float(s_mask.min()), float(s_mask.max())],
            dtype=float,
        )

        basins.append(
            Basin1D(
                id=basin_id,
                minimum=min_pos,
                f_min=f_min,
                radius=radius,
                bounds=bounds,
            )
        )

    if verbose:
        print(f"[1D basin detection] Found {len(basins)} basins.")
        for b in basins:
            print(
                f"  Basin {b.id}: min at {b.minimum:.3f}, "
                f"F_min={b.f_min:.3f}, radius≈{b.radius:.3f}, "
                f"bounds={b.bounds}"
            )

    return BasinNetwork1D(basins=basins, s=s, U=F, labels=labels)


def build_basin_network_from_potential_1d(
    potential,
    xlim=(-2.0, 2.0),
    ns=200,
    max_basins=None,
    verbose=True,
):
    """
    Sample a 1D potential on a grid and build a BasinNetwork1D.

    potential : callable x->[U, F]
        1D potential taking x=[s] or x=(s,) and returning (U, F).
    """
    xs = np.linspace(xlim[0], xlim[1], ns)
    U = np.empty(ns, dtype=float)
    for i, xv in enumerate(xs):
        u, _ = potential([xv])
        U[i] = u

    return build_basin_network_from_fes_1d(xs, U, max_basins=max_basins, verbose=verbose)


def detect_basins_for_mfpt_1d(
    potential,
    xlim=(-2.0, 2.0),
    ns=200,
    max_basins=None,
    verbose=True,
    core_fraction: Optional[float] = None,
    core_cut: Optional[float] = None,
):
    """
    Convenience wrapper for MFPT analysis in 1D.

    Samples the potential on a 1D grid, finds minima,
    and returns a BasinNetwork1D instance.
    """
    if core_fraction is not None or core_cut is not None:
        warnings.warn(
            "detect_basins_for_mfpt_1d() ignores legacy core_fraction/core_cut "
            "arguments in 1D; using full basin labels from the sampled potential.",
            UserWarning,
            stacklevel=2,
        )

    return build_basin_network_from_potential_1d(
        potential,
        xlim=xlim,
        ns=ns,
        max_basins=max_basins,
        verbose=verbose,
    )




def build_core_labels_from_full_labels(
    U_grid: np.ndarray,
    labels_full: np.ndarray,
    *,
    core_fraction: float = 0.05,
    core_cut: Optional[float] = None,
) -> np.ndarray:
    """Build *core* basin labels from a full partition.

    This is meant for kinetics (committors / MFPT / CTMC) where basins should be
    small *core sets* separated by an unlabeled transition region.

    Rules per basin b:
      - If core_cut is provided (in the same units as U_grid):
            core = {cells with labels_full==b and U <= Umin(b) + core_cut}
      - Else (default):
            core = lowest `core_fraction` energies within basin b.

    All non-core cells are labeled as -1.

    Guarantees at least one core cell per basin (the minimum cell).
    """
    U = np.asarray(U_grid, dtype=float)
    lab = np.asarray(labels_full, dtype=int)

    core = -np.ones_like(lab, dtype=int)

    finite = np.isfinite(U)
    basin_ids = np.unique(lab[lab >= 0])

    # Clamp fraction to sensible values
    f = float(core_fraction)
    if not (0.0 < f <= 1.0):
        raise ValueError('core_fraction must be in (0, 1]')

    for b in basin_ids:
        mask_b = (lab == b) & finite
        if not np.any(mask_b):
            continue

        Ub = U[mask_b]
        Umin = float(np.nanmin(Ub))

        if core_cut is not None:
            thr = Umin + float(core_cut)
        else:
            # Use per-basin quantile (unit-free)
            thr = float(np.nanquantile(Ub, f))

        mask_core = mask_b & (U <= thr)

        # Ensure at least one cell: include the minimum cell(s)
        if not np.any(mask_core):
            mask_core = mask_b & (U == Umin)

        core[mask_core] = int(b)

    return core

# ---------- high-level: automatic basin detection ----------

def build_basin_network_from_potential(
    potential,
    xlim=(-2.0, 2.0),
    ylim=(-2.0, 2.0),
    nx=200,
    ny=200,
    max_basins=None,
    verbose=True,
    core_fraction: float = 0.05,
    core_cut: Optional[float] = None,
):
    """
    Automatic detection of basins of attraction for MFPT analysis.

    1) Samples U(x) on a grid.
    2) Finds local minima.
    3) Assigns each grid point to the nearest minimum in (x,y) space.
    4) Builds Basin objects with rough radius and bounds.

    Parameters
    ----------
    potential : callable x->[U, F]
        Analytic potential or FESPotential.
    xlim, ylim : tuple of floats
        Limits of the grid in x and y.
    nx, ny : int
        Number of grid points along x and y.
    max_basins : int or None
        If not None, keep only the lowest-energy max_basins minima.

    Returns
    -------
    BasinNetwork
    """
    xs, ys, U = sample_potential_grid(potential, xlim=xlim, ylim=ylim, nx=nx, ny=ny)

    minima_idx = _find_local_minima_grid(U, max_basins=max_basins, sort_by_energy=True)
    if minima_idx.size == 0:
        raise RuntimeError("No local minima found on the grid; check bounds/resolution.")

    labels = _assign_labels_by_nearest_minima(xs, ys, minima_idx)

    basins = []
    for basin_id, (i_min, j_min) in enumerate(minima_idx):
        mask = labels == basin_id
        if not np.any(mask):
            continue

        f_min = float(U[i_min, j_min])

        coords = np.column_stack(np.nonzero(mask))  # [[i,j], ...]
        # Physical coordinates of basin grid points
        xs_mask = xs[coords[:, 0]]
        ys_mask = ys[coords[:, 1]]

        dx = xs_mask - xs[i_min]
        dy = ys_mask - ys[j_min]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        radius = float(dist.max())

        bounds = np.array(
            [
                [float(xs_mask.min()), float(xs_mask.max())],
                [float(ys_mask.min()), float(ys_mask.max())],
            ]
        )

        basins.append(
            Basin(
                id=basin_id,
                minimum=np.array([xs[i_min], ys[j_min]], dtype=float),
                f_min=f_min,
                radius=radius,
                bounds=bounds,
            )
        )

    # Any NaN/inf regions get label -1
    labels[~np.isfinite(U)] = -1


    labels_full = labels.copy()
    core_labels = build_core_labels_from_full_labels(U, labels_full, core_fraction=core_fraction, core_cut=core_cut)
    if verbose:
        print(f"[Basin detection] Found {len(basins)} basins.")
        for b in basins:
            print(
                f"  Basin {b.id}: min at {b.minimum}, "
                f"F_min={b.f_min:.3f}, radius≈{b.radius:.3f}"
            )

    return BasinNetwork(basins=basins, xs=xs, ys=ys, U=U, labels=labels, labels_full=labels_full, core_labels=core_labels)


def detect_basins_for_mfpt(
    potential,
    xlim=(-2.0, 2.0),
    ylim=(-2.0, 2.0),
    nx=200,
    ny=200,
    max_basins=None,
    verbose=True,
    core_fraction: float = 0.05,
    core_cut: Optional[float] = None,
):
    """
    Convenience wrapper specifically for MFPT analysis:

    Returns:
        basin_network : BasinNetwork
    """
    return build_basin_network_from_potential(
        potential,
        xlim=xlim,
        ylim=ylim,
        nx=nx,
        ny=ny,
        max_basins=max_basins,
        verbose=verbose,
        core_fraction=core_fraction,
        core_cut=core_cut,
    )
