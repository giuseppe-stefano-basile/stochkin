"""stochkin.committor
===================

Committor (splitting probability) analysis.

The committor :math:`q(x)` is the probability that a trajectory started
at *x* reaches basin **B** before basin **A**.  This module provides
two complementary approaches:

1. **Trajectory-based (shooting) committor** – launch many short
   trajectories from each grid point and count outcomes.
   Works for both underdamped (Langevin) and overdamped (Brownian)
   dynamics.  Suitable for 1D and 2D grids.

   - :func:`run_short_trajectory` – single trajectory worker.
   - :func:`committor_point` – estimate *q* at one point (many trials).
   - :func:`committor_map_parallel` – 2D grid committor with multiprocessing.
   - :func:`committor_profile_1d` – 1D committor profile.

2. **FPE-based (grid) committor** – solve the backward Fokker–Planck
   equation :math:`L q = 0` with Dirichlet boundary conditions on a
   discrete grid built from a :class:`~stochkin.potentials.BasinNetwork`.

   - :func:`committor_map_fpe` – solve for *q(x, y)* via sparse linear algebra.

Example basin functions (:func:`basinA`, :func:`basinB`) for the
central-well / ring potential are included for quick prototyping.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

# Optional SciPy sparse solver
try:
    import scipy.sparse.linalg as _spla  # type: ignore

    _HAVE_SCIPY_SPARSE_LINALG = True
except ImportError:  # pragma: no cover - optional
    _HAVE_SCIPY_SPARSE_LINALG = False

from .fpe import build_fp_generator_from_fes
from .potentials import BasinNetwork
from .integrators import GammaToDiffusion, overdamped_step_euler, baobab_step
from .boundaries import apply_bounds as _apply_bounds, reflect_position_velocity as _reflect_xv

def run_short_trajectory(
    potential,
    x0,
    v0,
    max_time,
    dt,
    gamma,
    kT,
    basinA,
    basinB,
    m=1.0,
    regime="underdamped",
    diffusion=None,
    overdamped_eps=1e-6,
    bounds=None,
    boundary=None,
):
    """Run a single short trajectory and report which basin is hit first.

    This is the inner workhorse of the shooting-committor algorithm.
    Starting from ``(x0, v0)``, the trajectory is integrated until it
    enters *basinA* (returns ``'A'``), *basinB* (returns ``'B'``), or
    *max_time* is exceeded (returns ``None``).

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)`` with ``F = −∇U``.
    x0 : array_like
        Initial position.
    v0 : array_like
        Initial velocity (ignored for overdamped dynamics).
    max_time : float
        Maximum simulation time.
    dt : float
        Integration time step.
    gamma : float
        Friction coefficient.
    kT : float
        Thermal energy.
    basinA, basinB : callable
        ``basin(x) -> bool`` indicating membership.
    m : float
        Mass (default 1).
    regime : {'underdamped', 'overdamped'}
        Dynamics type.
    diffusion : scalar, callable, or None
        Diffusion coefficient for overdamped mode.
    overdamped_eps : float
        Finite-difference step for ∇D.
    bounds : sequence of (lo, hi) or None
        Bounding box for position enforcement.
    boundary : {'reflect', 'clip'} or None
        How to enforce *bounds*.

    Returns
    -------
    str or None
        ``'A'``, ``'B'``, or ``None`` if neither basin was reached.
    """
    x = np.array(x0, dtype=float).ravel()
    v = np.array(v0, dtype=float).ravel()
    t = 0.0

    # If bounds are provided but boundary isn't, default to reflecting.
    if bounds is not None and boundary is None:
        boundary = "reflect"

    # Enforce bounds at t=0 so we don't start outside the domain.
    if bounds is not None and boundary is not None:
        if str(boundary) == "reflect":
            if str(regime) == "overdamped":
                x = _apply_bounds(x, bounds, mode="reflect")
            else:
                x, v = _reflect_xv(x, v, bounds)
        else:
            x = _apply_bounds(x, bounds, mode=str(boundary))

    if regime == "overdamped":
        beta = 1.0 / kT
        if diffusion is None:
            diffusion = GammaToDiffusion(gamma=gamma, kT=kT)

        while t < max_time:
            x = overdamped_step_euler(
                potential, x, dt, beta, diffusion, eps=float(overdamped_eps)
            )
            if bounds is not None and boundary is not None:
                x = _apply_bounds(x, bounds, mode=str(boundary))
            if basinA(x): return "A"
            if basinB(x): return "B"
            t += dt
        return None

    while t < max_time:
        x, v, _ = baobab_step(potential, x, v, dt, gamma, kT, m)
        if bounds is not None and boundary is not None:
            if str(boundary) == "reflect":
                x, v = _reflect_xv(x, v, bounds)
            else:
                x = _apply_bounds(x, bounds, mode=str(boundary))
        if basinA(x): return "A"
        if basinB(x): return "B"
        t += dt

    return None

def committor_point(args):
    """Estimate the committor at a single grid point via trajectory shooting.

    Runs *n_trials* short trajectories from ``(x0, v0)`` and returns
    the fraction that reach basin B first:
    :math:`q = n_B / (n_A + n_B)`.  Returns ``NaN`` if no trajectory
    reached either basin.

    Parameters
    ----------
    args : tuple
        Positional arguments packed as a tuple for compatibility with
        ``multiprocessing.Pool.map``.  Fields (by index)::

            0  potential      5  n_trials     10  m
            1  x0             6  max_time     11  regime (optional)
            2  v0             7  dt           12  diffusion (optional)
            3  basinA         8  gamma        13  overdamped_eps (optional)
            4  basinB         9  kT           14  bounds (optional)
                                              15  boundary (optional)
                                              16  seed (optional)

    Returns
    -------
    float
        Estimated committor *q ∈ [0, 1]* (or ``NaN``).
    """
    # Backwards compatible positional parsing: older code passes 11 fields.
    potential = args[0]
    x0 = args[1]
    v0 = args[2]
    basinA = args[3]
    basinB = args[4]
    n_trials = int(args[5])
    max_time = float(args[6])
    dt = float(args[7])
    gamma = float(args[8])
    kT = float(args[9])
    m = float(args[10])

    regime = args[11] if len(args) > 11 else "underdamped"
    diffusion = args[12] if len(args) > 12 else None
    overdamped_eps = float(args[13]) if len(args) > 13 else 1e-6
    bounds = args[14] if len(args) > 14 else None
    boundary = args[15] if len(args) > 15 else None
    seed = args[16] if len(args) > 16 else None

    if seed is not None:
        np.random.seed(int(seed) % (2**32 - 1))

    # Run trials and count outcomes without storing all per-trial results.
    nA = 0
    nB = 0
    for _ in range(n_trials):
        r = run_short_trajectory(
            potential,
            x0,
            v0,
            max_time,
            dt,
            gamma,
            kT,
            basinA,
            basinB,
            m=m,
            regime=regime,
            diffusion=diffusion,
            overdamped_eps=overdamped_eps,
            bounds=bounds,
            boundary=boundary,
        )
        if r == "A":
            nA += 1
        elif r == "B":
            nB += 1
    return nB / (nA + nB) if (nA + nB) > 0 else np.nan


def committor_map_parallel(
    potential,
    basinA,
    basinB,
    xlim=(-1.5, 1.5),
    ylim=(-1.5, 1.5),
    grid_size=50,
    n_trials=100,
    max_time=50.0,
    dt=0.01,
    gamma=10.0,
    kT=0.05,
    m=1.0,
    processes=None,
    regime="underdamped",
    diffusion=None,
    overdamped_eps=1e-6,
    bounds=None,
    boundary=None,
    base_seed=None,
):
    """Compute a 2D committor map using parallel trajectory shooting.

    For each point on a ``grid_size × grid_size`` grid, *n_trials*
    short trajectories are launched and the fraction reaching basin B
    first is recorded.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)``.
    basinA, basinB : callable
        ``basin(x) -> bool``.
    xlim, ylim : tuple
        Grid bounds.
    grid_size : int
        Number of grid points per axis.
    n_trials : int
        Shooting trials per grid point.
    max_time : float
        Maximum trajectory time.
    dt : float
        Time step.
    gamma : float
        Friction.
    kT : float
        Thermal energy.
    m : float
        Mass (default 1).
    processes : int or None
        Number of worker processes (None = all CPUs).
    regime : {'underdamped', 'overdamped'}
        Dynamics type.
    diffusion : scalar, callable, or None
        Diffusion for overdamped mode.
    overdamped_eps : float
        FD step for ∇D.
    bounds : sequence of (lo, hi) or None
        Bounding box.
    boundary : {'reflect', 'clip'} or None
        Bound enforcement mode.
    base_seed : int or None
        Base seed for reproducible per-point seeds.

    Returns
    -------
    xs : ndarray
        x-axis grid.
    ys : ndarray
        y-axis grid.
    Q : ndarray, shape (grid_size, grid_size)
        Committor values *q(x, y) ∈ [0, 1]*.
    """
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)

    if bounds is not None and boundary is None:
        boundary = "reflect"

    rng = np.random.RandomState(int(base_seed)) if base_seed is not None else None

    args = []
    for xi in xs:
        for yj in ys:
            args.append(
                (
                    potential,
                    [xi, yj],
                    [0.0, 0.0],
                    basinA,
                    basinB,
                    n_trials,
                    max_time,
                    dt,
                    gamma,
                    kT,
                    m,
                    regime,
                    diffusion,
                    overdamped_eps,
                    bounds,
                    boundary,
                    (int(rng.randint(0, 2**31 - 1)) if rng is not None else None),
                )
            )

    n_tasks = len(args)
    if n_tasks == 0:
        return xs, ys, np.empty((0, 0), dtype=float)

    if processes is None:
        n_procs = int(os.cpu_count() or 1)
    else:
        n_procs = int(processes)
    n_procs = max(1, min(n_procs, n_tasks))

    if n_procs <= 1:
        results = [committor_point(a) for a in args]
    else:
        pool = Pool(processes=n_procs)
        try:
            chunksize = max(1, n_tasks // max(1, n_procs * 8))
            # Use map (ordered) so reshape aligns with the (xi, yj) loop order.
            results = pool.map(committor_point, args, chunksize=chunksize)
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
            pool.join()
            raise

    Q = np.array(results).reshape((grid_size, grid_size))

    plt.figure(figsize=(6, 5))
    plt.imshow(
        Q.T,
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    plt.colorbar(label="Committor q(x)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Grid-based committor map (parallel)")
    plt.show()

    return xs, ys, Q

def committor_profile_1d(
    potential,
    basinA,
    basinB,
    xlim=(-1.5, 1.5),
    grid_size=50,
    n_trials=100,
    max_time=50.0,
    dt=0.01,
    gamma=10.0,
    kT=0.05,
    m=1.0,
):
    """Compute the committor q(x) on a 1D grid via short trajectory shooting.

    This is the 1D analogue of :func:`committor_map_parallel`.

    Parameters are the same as in :func:`committor_map_parallel`, but with a 1D grid.

    Returns
    -------
    xs : (grid_size,) ndarray
        Grid points.
    q_vals : (grid_size,) ndarray
        Committor values in [0, 1].
    """
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    q_vals = np.empty_like(xs, dtype=float)

    for i, xv in enumerate(xs):
        args = (
            potential,
            np.array([xv], dtype=float),
            np.array([0.0], dtype=float),
            basinA,
            basinB,
            int(n_trials),
            float(max_time),
            float(dt),
            float(gamma),
            float(kT),
            float(m),
        )
        q_vals[i] = committor_point(args)

    return xs, q_vals

# Example basins for a central-well/ring potential

def basinA(x, rA=0.1):
    """Basin A: central well around the origin."""
    return np.linalg.norm(x) < rA


def basinB(x, r_inner=0.90, r_outer=1.0):
    """Basin B: circular corona (ring) between r_inner and r_outer."""
    r = np.linalg.norm(x)
    return (r_inner <= r <= r_outer)

def committor_map_fpe(
    basin_network: BasinNetwork,
    D,
    beta,
    basinA_id,
    basinB_id,
    sparse=True,
    energy_buffer_kT=None,
):
    """Compute the committor *q(x, y)* by solving the backward FPE on a grid.

    Solves :math:`L q = 0` with Dirichlet conditions
    :math:`q = 0` on basin A and :math:`q = 1` on basin B, where *L*
    is the detailed-balance-preserving discrete Fokker–Planck generator
    built from the free-energy surface.

    Parameters
    ----------
    basin_network : BasinNetwork
        Must provide ``xs``, ``ys``, ``U``, and ``labels``.
    D : float or ndarray, shape (nx, ny)
        Diffusion coefficient (scalar or field).
    beta : float
        Inverse temperature :math:`1/(k_BT)`.
    basinA_id, basinB_id : int
        Integer basin labels for the two absorbing states.
    sparse : bool
        Use sparse linear algebra (default ``True``).
    energy_buffer_kT : float or None
        If given, restrict A and B to low-energy cores within
        *energy_buffer_kT* of the basin minimum (in kT units).
        The remainder of each basin becomes part of the solved domain.

    Returns
    -------
    xs, ys : ndarray
        Grid coordinate arrays.
    q_grid : ndarray, shape (nx, ny)
        Committor values in [0, 1].

    Raises
    ------
    ImportError
        If SciPy sparse is not available.
    """
    if not _HAVE_SCIPY_SPARSE_LINALG:
        raise ImportError(
            "committor_map_fpe requires scipy.sparse.linalg. "
            "Install SciPy or use the trajectory-based committor_map_parallel."
        )

    xs = np.asarray(basin_network.xs, dtype=float)
    ys = np.asarray(basin_network.ys, dtype=float)
    U = np.asarray(basin_network.U, dtype=float)
    labels = np.asarray(basin_network.labels, dtype=int)

    nx, ny = U.shape
    N = nx * ny

    U_flat = U.reshape(N)
    labels_flat = labels.reshape(N)

    # 1) Build initial A/B masks based purely on labels
    labelA_mask = labels_flat == basinA_id
    labelB_mask = labels_flat == basinB_id

    if not labelA_mask.any():
        raise ValueError(f"No grid points found with basinA_id={basinA_id}.")
    if not labelB_mask.any():
        raise ValueError(f"No grid points found with basinB_id={basinB_id}.")

    # 2) Optionally restrict to low-energy cores inside each basin
    if energy_buffer_kT is not None:
        # local energy thresholds in each basin
        U_A = U_flat[labelA_mask]
        U_B = U_flat[labelB_mask]

        U_A_min = U_A.min()
        U_B_min = U_B.min()

        # convert buffer in kT (dimensionless) to energy units
        buffer_energy = energy_buffer_kT / beta

        A_mask = labelA_mask & (U_flat <= U_A_min + buffer_energy)
        B_mask = labelB_mask & (U_flat <= U_B_min + buffer_energy)

        # Fallback if thresholds are too strict and we emptied a basin
        if not A_mask.any():
            A_mask = labelA_mask
        if not B_mask.any():
            B_mask = labelB_mask
    else:
        # Use the full basins as boundary sets
        A_mask = labelA_mask
        B_mask = labelB_mask

    # 3) Determine unknown region (where we solve the PDE)
    unknown_mask = ~(A_mask | B_mask)
    unknown_idx = np.where(unknown_mask)[0]

    # If there are no unknown states, the committor is trivial:
    #  - q=0 in A, q=1 in B
    if unknown_idx.size == 0:
        q_flat = np.zeros(N, dtype=float)
        q_flat[A_mask] = 0.0
        q_flat[B_mask] = 1.0
        q_grid = q_flat.reshape(nx, ny)
        return xs, ys, q_grid

    # 4) Build FP generator
    L = build_fp_generator_from_fes(xs, ys, U, D, beta, sparse=sparse)

    # 5) Restrict generator to unknown states
    A_sub = L[unknown_idx, :][:, unknown_idx]

    # Right-hand side: b_i = - Σ_{j∈B} L_ij * 1 (q_B = 1), q_A = 0
    B_idx = np.where(B_mask)[0]
    if B_idx.size > 0:
        contrib = L[unknown_idx, :][:, B_idx].sum(axis=1)
        b = -np.asarray(contrib).ravel()
    else:
        b = np.zeros(unknown_idx.size, dtype=float)

    # 6) Solve for q on unknown states
    q_unknown = _spla.spsolve(A_sub, b)

    # 7) Assemble full committor
    q_flat = np.zeros(N, dtype=float)
    q_flat[A_mask] = 0.0
    q_flat[B_mask] = 1.0
    q_flat[unknown_idx] = q_unknown

    q_grid = q_flat.reshape(nx, ny)

    return xs, ys, q_grid
