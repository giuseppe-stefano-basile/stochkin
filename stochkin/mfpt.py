"""stochkin.mfpt
==============

Mean first-passage time (MFPT) and transition-rate estimation.

This module provides tools for computing MFPTs between metastable
basins on a free-energy landscape:

**Two-basin (pairwise) MFPT**

- :func:`compute_mfpt` / :func:`compute_mfpt_1d` – trajectory-shooting
  MFPT between two basins (Langevin or overdamped BD).
- :func:`compute_bidirectional_mfpt` / :func:`compute_bidirectional_mfpt_1d`
  – convenience wrappers that compute A→B and B→A in one call.

**Multi-basin MFPT network**

- :func:`compute_mfpt_network` – first-exit overdamped BD simulations
  across all basins of a :class:`~stochkin.potentials.BasinNetwork`.
- :func:`compute_mfpt_network_fpe` – grid-based backward Kolmogorov
  solve for all-pairs MFPTs (no trajectories needed).

**CTMC rate matrix estimation**

- :func:`estimate_rate_matrix` – build a continuous-time Markov chain
  generator :math:`K` from first-exit statistics or inverse-MFPT
  fallback.

**Diagnostics**

- :func:`estimate_transition_rates` – simple two-state rate and
  equilibrium-population summary.
- :func:`plot_mfpt_statistics` – histogram of passage-time
  distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
import warnings
from tqdm import tqdm
from .potentials import BasinNetwork  # for type / logic

# Optional SciPy sparse solver
try:
    import scipy.sparse.linalg as _spla  # type: ignore

    _HAVE_SCIPY_SPARSE_LINALG = True
except ImportError:  # pragma: no cover - optional
    _HAVE_SCIPY_SPARSE_LINALG = False

from .fpe import build_fp_generator_from_fes


# =========================
# Existing two-basin MFPT
# =========================

from .integrators import baobab_step, overdamped_step_euler, GammaToDiffusion
from .boundaries import apply_bounds as _apply_bounds, reflect_position_velocity as _reflect_xv


def _km_restricted_mean(event_times, n_total, t_max):
    """Kaplan–Meier restricted mean E[min(T, t_max)] with right-censoring at t_max.

    Parameters
    ----------
    event_times : array-like
        Observed exit/event times (successful exits only).
    n_total : int
        Total number of trials started in the basin (events + censored).
    t_max : float
        Censoring time (max_time).

    Returns
    -------
    rmean : float
        Restricted mean E[min(T, t_max)] = ∫_0^{t_max} S(t) dt.
        This is a *lower bound* on E[T].
    S_tmax : float
        Estimated survival probability S(t_max-): fraction of trials not exited
        strictly before t_max.
    """
    n_total = int(n_total)
    t_max = float(t_max)
    if n_total <= 0:
        return np.nan, np.nan
    if t_max <= 0.0:
        return 0.0, 1.0

    times = np.asarray(event_times, dtype=float).reshape(-1)
    if times.size == 0:
        return float(t_max), 1.0

    times = times[np.isfinite(times)]
    times = times[(times >= 0.0) & (times <= t_max)]
    if times.size == 0:
        return float(t_max), 1.0

    times.sort()
    uniq, counts = np.unique(times, return_counts=True)

    S = 1.0
    n_at_risk = n_total
    prev_t = 0.0
    area = 0.0

    for t, d in zip(uniq, counts):
        t = float(t)
        if t < prev_t:
            continue
        if t > t_max:
            break
        area += S * (t - prev_t)
        if n_at_risk <= 0:
            prev_t = t
            break
        d = int(d)
        if d > n_at_risk:
            d = n_at_risk
        S *= (1.0 - float(d) / float(n_at_risk))
        n_at_risk -= d
        prev_t = t

    area += S * (t_max - prev_t)
    return float(area), float(S)

def single_passage_time(args):
    """Run a single trajectory and return the first-passage time.

    This is a top-level, picklable worker designed for use with
    ``multiprocessing.Pool.map``.  It integrates the dynamics from a
    starting position inside *start_basin* until *target_basin* is
    reached (returns the elapsed time) or *max_time* is exceeded
    (returns ``None``).

    Parameters
    ----------
    args : tuple
        Packed positional arguments (for multiprocessing compatibility):

        0. potential : callable – ``(U, F) = potential(x)``
        1. start_basin : callable – ``bool = start_basin(x)``
        2. target_basin : callable – ``bool = target_basin(x)``
        3. x0 : array_like – initial position
        4. v0 : array_like – initial velocity
        5. max_time : float – maximum simulation time
        6. dt : float – integration time-step
        7. gamma : float – friction
        8. kT : float – thermal energy
        9. m : float – mass
        10. regime : {'underdamped', 'overdamped'}
        11. diffusion : scalar, callable, or None

        Optional (appended for newer code):

        12. bounds : sequence of (lo, hi) or None
        13. boundary : {'reflect', 'clip'} or None
        14. seed : int or None – per-trajectory RNG seed
        15. overdamped_eps : float – FD step for ∇D

    Returns
    -------
    float or None
        First-passage time, or ``None`` if *max_time* was exceeded.
    """
    # Required (historical) fields
    (
        potential,
        start_basin,
        target_basin,
        x0,
        v0,
        max_time,
        dt,
        gamma,
        kT,
        m,
        regime,
        diffusion,
    ) = args[:12]

    bounds = args[12] if len(args) > 12 else None
    boundary = args[13] if len(args) > 13 else None
    seed = args[14] if len(args) > 14 else None
    overdamped_eps = float(args[15]) if len(args) > 15 else 1e-6

    if seed is not None:
        np.random.seed(int(seed) % (2**32 - 1))

    x = np.array(x0, dtype=float).ravel()
    v = np.array(v0, dtype=float).ravel()

    # Enforce bounds at t=0 so we don't start outside the domain
    if bounds is not None and boundary is not None:
        if str(boundary) == "reflect":
            # in overdamped, v is irrelevant; in underdamped we flip it consistently
            if str(regime) == "overdamped":
                x = _apply_bounds(x, bounds, mode="reflect")
            else:
                x, v = _reflect_xv(x, v, bounds)
        else:
            x = _apply_bounds(x, bounds, mode=str(boundary))

    t = 0.0

    if not start_basin(x):
        return None

    if regime == "overdamped":
        beta = 1.0 / float(kT)
        if diffusion is None:
            diffusion = GammaToDiffusion(gamma=gamma, kT=kT)

        while t < max_time:
            x = overdamped_step_euler(potential, x, dt, beta, diffusion, eps=overdamped_eps)
            if bounds is not None and boundary is not None:
                x = _apply_bounds(x, bounds, mode=str(boundary))
            t += dt
            if target_basin(x):
                return t

        return None

    # underdamped
    while t < max_time:
        x, v, _ = baobab_step(potential, x, v, dt, gamma, kT, m)
        if bounds is not None and boundary is not None:
            if str(boundary) == "reflect":
                x, v = _reflect_xv(x, v, bounds)
            else:
                x = _apply_bounds(x, bounds, mode=str(boundary))
        t += dt
        if target_basin(x):
            return t

    return None


def generate_basin_position(
    basin_func,
    bounds,
    max_attempts=1000,
):
    """Sample a random 2D position inside a basin (rejection sampling).

    Parameters
    ----------
    basin_func : callable
        ``basin_func(pos) -> bool`` indicating membership.
    bounds : array_like, shape (2, 2)
        ``[[xmin, xmax], [ymin, ymax]]`` bounding box.
    max_attempts : int
        Maximum rejection-sampling iterations.

    Returns
    -------
    ndarray of shape (2,) or None
        A point inside the basin, or ``None`` if sampling failed.
    """
    for _ in range(max_attempts):
        x = np.random.uniform(bounds[0][0], bounds[0][1])
        y = np.random.uniform(bounds[1][0], bounds[1][1])
        pos = np.array([x, y])
        if basin_func(pos):
            return pos
    return None


def generate_basin_position_1d(
    basin_func,
    bounds,
    max_attempts=1000,
):
    """
    1D variant of generate_basin_position for x in [xmin, xmax].

    Parameters
    ----------
    basin_func : callable
        Returns True if position pos = np.array([x]) is in the basin.
    bounds : (xmin, xmax)
        Interval from which to sample candidate x.
    max_attempts : int
        Maximum attempts to find a valid position.

    Returns
    -------
    np.ndarray of shape (1,) or None if no valid position found.
    """
    xmin, xmax = float(bounds[0]), float(bounds[1])
    for _ in range(max_attempts):
        x = np.random.uniform(xmin, xmax)
        pos = np.array([x], dtype=float)
        if basin_func(pos):
            return pos
    return None


def compute_mfpt(
    potential,
    start_basin,
    target_basin,
    n_trials=1000,
    max_time=1000.0,
    dt=0.01,
    gamma=10.0,
    kT=0.05,
    m=1.0,
    start_bounds=None,
    initial_velocity=(0.0, 0.0),
    processes=None,
    verbose=True,
    regime="underdamped",
    diffusion=None,
    # New: bounds + reproducible RNG + overdamped FD control
    overdamped_eps=1e-6,
    bounds=None,
    boundary=None,
    base_seed=None,
):
    """Compute the mean first-passage time from *start_basin* to *target_basin*.

    Launches *n_trials* independent trajectories (Langevin or overdamped BD)
    from random positions inside *start_basin* and records the time of first
    entry into *target_basin*.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)``.
    start_basin, target_basin : callable
        ``basin(x) -> bool``.
    n_trials : int
        Number of independent trajectories.
    max_time : float
        Maximum simulation time per trajectory.
    dt : float
        Integration time step.
    gamma : float
        Friction coefficient.
    kT : float
        Thermal energy.
    m : float
        Mass.
    start_bounds : sequence of (lo, hi) or None
        Bounding box for initial-position sampling.
    initial_velocity : array_like or ``'thermal'``
        Fixed initial velocity vector, or ``'thermal'`` to draw from
        the Maxwell–Boltzmann distribution.
    processes : int or None
        Worker processes (None = single process).
    verbose : bool
        Print progress.
    regime : {'underdamped', 'overdamped'}
        Dynamics type.
    diffusion : scalar, callable, or None
        Diffusion coefficient for overdamped mode.
    overdamped_eps : float
        FD step for ∇D.
    bounds : sequence of (lo, hi) or None
        Domain bounds (essential for cropped-FES workflows).
    boundary : {'reflect', 'clip'} or None
        Bound enforcement method.
    base_seed : int or None
        Base seed for reproducible per-trajectory seeds.

    Returns
    -------
    dict
        Keys: ``'mean'``, ``'std'``, ``'successful_trials'``,
        ``'passage_times'`` (list), ``'success_rate'``,
        ``'total_trials'``.
    """
    if bounds is not None and boundary is None:
        boundary = "reflect"

    rng = np.random.RandomState(int(base_seed)) if base_seed is not None else None

    def _sample_start():
        # deterministic if rng is not None
        if start_bounds is None:
            x0 = np.array([0.0, 0.0], dtype=float)
            return x0 if start_basin(x0) else None

        # expect start_bounds = ((xmin,xmax),(ymin,ymax)) or similar
        b0, b1 = start_bounds[0], start_bounds[1]
        xmin, xmax = float(b0[0]), float(b0[1])
        ymin, ymax = float(b1[0]), float(b1[1])

        for _ in range(10000):
            if rng is None:
                x0 = np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)], dtype=float)
            else:
                x0 = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float)
            if start_basin(x0):
                return x0
        return None

    args_list = []
    successful_starts = 0

    for _ in range(int(n_trials)):
        x0 = _sample_start()
        if x0 is None:
            continue

        if isinstance(initial_velocity, str) and initial_velocity == "thermal":
            dim = len(np.asarray(x0, dtype=float).ravel())
            if rng is None:
                v0 = np.random.normal(0.0, np.sqrt(float(kT) / float(m)), dim)
            else:
                v0 = rng.normal(0.0, np.sqrt(float(kT) / float(m)), dim)
        else:
            v0 = np.array(initial_velocity, dtype=float).ravel()

        seed = int(rng.randint(0, 2**31 - 1)) if rng is not None else None

        args_list.append(
            (
                potential,
                start_basin,
                target_basin,
                x0,
                v0,
                float(max_time),
                float(dt),
                float(gamma),
                float(kT),
                float(m),
                str(regime),
                diffusion,
                bounds,
                boundary,
                seed,
                float(overdamped_eps),
            )
        )
        successful_starts += 1

    if verbose:
        print(f"Starting {successful_starts} trajectories...")

    if successful_starts == 0:
        if verbose:
            print("Warning: No valid initial conditions found in the start basin!")
        return {
            "mean": np.nan,
            "std": np.nan,
            "successful_trials": 0,
            "passage_times": [],
            "success_rate": 0.0,
            "total_trials": 0,
        }

    if processes == 1:
        passage_times = [single_passage_time(args) for args in args_list]
    else:
        with Pool(processes=processes) as pool:
            passage_times = pool.map(single_passage_time, args_list)

    successful_times = [t for t in passage_times if t is not None]
    n_successful = len(successful_times)

    if n_successful == 0:
        if verbose:
            print("Warning: No successful transitions found!")
        return {
            "mean": np.nan,
            "std": np.nan,
            "successful_trials": 0,
            "passage_times": [],
            "success_rate": 0.0,
            "total_trials": successful_starts,
        }

    mean_fpt = float(np.mean(successful_times))
    std_fpt = float(np.std(successful_times))
    success_rate = n_successful / float(successful_starts)

    if verbose:
        print(
            f"Successful transitions: {n_successful}/{successful_starts} "
            f"({success_rate:.2%})"
        )
        print(f"Mean first passage time: {mean_fpt:.3f} ± {std_fpt:.3f}")

    return {
        "mean": mean_fpt,
        "std": std_fpt,
        "successful_trials": n_successful,
        "passage_times": successful_times,
        "success_rate": success_rate,
        "total_trials": successful_starts,
    }


def compute_mfpt_1d(
    potential,
    start_basin,
    target_basin,
    n_trials=1000,
    max_time=1000.0,
    dt=0.01,
    gamma=10.0,
    kT=0.05,
    m=1.0,
    start_bounds=None,
    initial_velocity=(0.0,),
    processes=None,
    verbose=True,
    regime="underdamped",
    diffusion=None,
    overdamped_eps=1e-6,
    bounds=None,
    boundary=None,
    base_seed=None,
):
    """1D analogue of :func:`compute_mfpt`.

    Identical interface to :func:`compute_mfpt` but initial positions
    are 1D scalars and *start_bounds* is a simple ``(xmin, xmax)``
    interval.  Supports domain bounds for cropped 1D free-energy
    surfaces.

    Returns
    -------
    dict
        Same keys as :func:`compute_mfpt`.
    """
    if bounds is not None and boundary is None:
        boundary = "reflect"

    rng = np.random.RandomState(int(base_seed)) if base_seed is not None else None

    def _sample_start():
        if start_bounds is None:
            x0 = np.array([0.0], dtype=float)
            return x0 if start_basin(x0) else None

        xmin, xmax = float(start_bounds[0]), float(start_bounds[1])
        for _ in range(10000):
            x = (rng.uniform(xmin, xmax) if rng is not None else np.random.uniform(xmin, xmax))
            pos = np.array([x], dtype=float)
            if start_basin(pos):
                return pos
        return None

    args_list = []
    successful_starts = 0

    for _ in range(int(n_trials)):
        x0 = _sample_start()
        if x0 is None:
            continue

        if isinstance(initial_velocity, str) and initial_velocity == "thermal":
            dim = 1
            sigma = np.sqrt(float(kT) / float(m))
            v0 = (rng.normal(0.0, sigma, dim) if rng is not None else np.random.normal(0.0, sigma, dim))
        else:
            v0 = np.array(initial_velocity, dtype=float).ravel()

        seed = int(rng.randint(0, 2**31 - 1)) if rng is not None else None

        args_list.append(
            (
                potential,
                start_basin,
                target_basin,
                x0,
                v0,
                float(max_time),
                float(dt),
                float(gamma),
                float(kT),
                float(m),
                str(regime),
                diffusion,
                bounds,
                boundary,
                seed,
                float(overdamped_eps),
            )
        )
        successful_starts += 1

    if verbose:
        print(f"Starting {successful_starts} trajectories...")

    if successful_starts == 0:
        if verbose:
            print("Warning: No valid initial conditions found in the start basin!")
        return {
            "mean": np.nan,
            "std": np.nan,
            "successful_trials": 0,
            "passage_times": [],
            "success_rate": 0.0,
            "total_trials": 0,
        }

    if processes == 1:
        passage_times = [single_passage_time(args) for args in args_list]
    else:
        with Pool(processes=processes) as pool:
            passage_times = pool.map(single_passage_time, args_list)

    successful_times = [t for t in passage_times if t is not None]
    n_successful = len(successful_times)

    if n_successful == 0:
        if verbose:
            print("Warning: No successful transitions found!")
        return {
            "mean": np.nan,
            "std": np.nan,
            "successful_trials": 0,
            "passage_times": [],
            "success_rate": 0.0,
            "total_trials": successful_starts,
        }

    mean_fpt = float(np.mean(successful_times))
    std_fpt = float(np.std(successful_times))
    success_rate = n_successful / float(successful_starts)

    if verbose:
        print(
            f"Successful transitions: {n_successful}/{successful_starts} "
            f"({success_rate:.2%})"
        )
        print(f"Mean first passage time: {mean_fpt:.3f} ± {std_fpt:.3f}")

    return {
        "mean": mean_fpt,
        "std": std_fpt,
        "successful_trials": n_successful,
        "passage_times": successful_times,
        "success_rate": success_rate,
        "total_trials": successful_starts,
    }


def compute_bidirectional_mfpt(
    potential,
    basinA,
    basinB,
    n_trials=500,
    max_time=1000.0,
    dt=0.01,
    gamma=10.0,
    kT=0.05,
    m=1.0,
    boundsA=None,
    boundsB=None,
    initial_velocity=(0.0, 0.0),
    processes=None,
    verbose=True,
    regime="underdamped",
    diffusion=None
):
    """Compute MFPT in both directions: A→B and B→A.

    Convenience wrapper that calls :func:`compute_mfpt` twice.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)``.
    basinA, basinB : callable
        Basin membership tests.
    n_trials : int
        Trajectories per direction.
    max_time, dt, gamma, kT, m : float
        Dynamics parameters.
    boundsA, boundsB : sequence of (lo, hi) or None
        Sampling boxes for each basin.
    initial_velocity : array_like or ``'thermal'``
        Shared initial-velocity setting.
    processes : int or None
        Worker processes.
    verbose : bool
        Print progress.
    regime : {'underdamped', 'overdamped'}
        Dynamics type.
    diffusion : scalar, callable, or None
        Diffusion for overdamped mode.

    Returns
    -------
    dict
        ``{'A_to_B': <mfpt_dict>, 'B_to_A': <mfpt_dict>}``.
    """
    if verbose:
        print("Computing MFPT from Basin A to Basin B...")
    mfpt_AtoB = compute_mfpt(
        potential,
        basinA,
        basinB,
        n_trials=n_trials,
        max_time=max_time,
        dt=dt,
        gamma=gamma,
        kT=kT,
        m=m,
        start_bounds=boundsA,
        initial_velocity=initial_velocity,
        processes=processes,
        verbose=verbose,
        regime=regime,
        diffusion=diffusion
    )

    if verbose:
        print("\nComputing MFPT from Basin B to Basin A...")
    mfpt_BtoA = compute_mfpt(
        potential,
        basinB,
        basinA,
        n_trials=n_trials,
        max_time=max_time,
        dt=dt,
        gamma=gamma,
        kT=kT,
        m=m,
        start_bounds=boundsB,
        initial_velocity=initial_velocity,
        processes=processes,
        verbose=verbose,
        regime=regime,
        diffusion=diffusion
    )

    return {"A_to_B": mfpt_AtoB, "B_to_A": mfpt_BtoA}


def compute_bidirectional_mfpt_1d(
    potential,
    basinA,
    basinB,
    n_trials=1000,
    max_time=1000.0,
    dt=0.01,
    gamma=10.0,
    kT=0.05,
    m=1.0,
    boundsA=None,
    boundsB=None,
    initial_velocity=0.0,
    processes=None,
    verbose=True,
    regime="underdamped",
    diffusion=None
):
    """1D analogue of :func:`compute_bidirectional_mfpt`.

    Computes MFPT in both directions (A→B and B→A) using
    :func:`compute_mfpt_1d`.

    Returns
    -------
    results_AB, results_BA : dict, dict
        Two MFPT result dicts as returned by :func:`compute_mfpt_1d`.
    """
    res_AB = compute_mfpt_1d(
        potential,
        start_basin=basinA,
        target_basin=basinB,
        n_trials=n_trials,
        max_time=max_time,
        dt=dt,
        gamma=gamma,
        kT=kT,
        m=m,
        start_bounds=boundsA,
        initial_velocity=initial_velocity,
        processes=processes,
        verbose=verbose,
        regime=regime,
        diffusion=diffusion
    )

    res_BA = compute_mfpt_1d(
        potential,
        start_basin=basinB,
        target_basin=basinA,
        n_trials=n_trials,
        max_time=max_time,
        dt=dt,
        gamma=gamma,
        kT=kT,
        m=m,
        start_bounds=boundsB,
        initial_velocity=initial_velocity,
        processes=processes,
        verbose=verbose,
        regime=regime,
        diffusion=diffusion
    )

    return res_AB, res_BA


def plot_mfpt_statistics(mfpt_results, title="MFPT Results"):
    """Plot histograms of passage-time distributions.

    Handles both single-direction results (one panel) and bidirectional
    results (two panels side by side).

    Parameters
    ----------
    mfpt_results : dict
        Output of :func:`compute_mfpt`, :func:`compute_mfpt_1d`, or
        :func:`compute_bidirectional_mfpt`.  If the dict contains
        ``'A_to_B'`` and ``'B_to_A'`` keys, two histograms are plotted.
    title : str
        Figure super-title.
    """
    if isinstance(mfpt_results, dict) and "A_to_B" in mfpt_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for i, (direction, data) in enumerate(mfpt_results.items()):
            if data["successful_trials"] > 0:
                axes[i].hist(
                    data["passage_times"],
                    bins=30,
                    alpha=0.7,
                    density=True,
                )
                axes[i].axvline(
                    data["mean"],
                    color="red",
                    linestyle="--",
                    label=f"Mean = {data['mean']:.3f}",
                )
                axes[i].set_xlabel("Passage Time")
                axes[i].set_ylabel("Density")
                axes[i].set_title(direction.replace("_", " → "))
                axes[i].legend()

        plt.suptitle(title)
        plt.tight_layout()
    else:
        if mfpt_results["successful_trials"] > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(
                mfpt_results["passage_times"],
                bins=30,
                alpha=0.7,
                density=True,
            )
            plt.axvline(
                mfpt_results["mean"],
                color="red",
                linestyle="--",
                label=f"Mean = {mfpt_results['mean']:.3f}",
            )
            plt.xlabel("First Passage Time")
            plt.ylabel("Density")
            plt.title(title)
            plt.legend()
        else:
            print("No successful transitions to plot.")

    plt.show()


def estimate_transition_rates(mfpt_results, verbose=True):
    """Estimate transition rates from MFPT results (simple two-state model).

    For a single direction, :math:`k = 1/\\tau`.  For bidirectional
    results, also computes the equilibrium constant
    :math:`K_{eq} = k_{AB}/k_{BA}` and the steady-state populations
    :math:`p_A, p_B`.

    Parameters
    ----------
    mfpt_results : dict
        Output of :func:`compute_mfpt` or :func:`compute_bidirectional_mfpt`.
    verbose : bool
        Print summary.

    Returns
    -------
    dict or None
        For bidirectional: ``{'k_AB', 'k_BA', 'K_eq', 'p_A_eq', 'p_B_eq'}``.
        For single direction: ``{'rate'}``.
        ``None`` if MFPTs are NaN.
    """
    if isinstance(mfpt_results, dict) and "A_to_B" in mfpt_results:
        tau_AB = mfpt_results["A_to_B"]["mean"]
        tau_BA = mfpt_results["B_to_A"]["mean"]

        if not (np.isnan(tau_AB) or np.isnan(tau_BA)):
            k_AB = 1.0 / tau_AB
            k_BA = 1.0 / tau_BA

            K_eq = k_AB / k_BA

            if verbose:
                print(f"Transition rate A→B: k_AB = {k_AB:.6f}")
                print(f"Transition rate B→A: k_BA = {k_BA:.6f}")
                print(f"Equilibrium constant K = k_AB/k_BA = {K_eq:.3f}")

                p_A = k_BA / (k_AB + k_BA)
                p_B = k_AB / (k_AB + k_BA)
                print(f"Equilibrium populations: P(A) = {p_A:.3f}, P(B) = {p_B:.3f}")

            return {
                "k_AB": k_AB,
                "k_BA": k_BA,
                "K_eq": K_eq,
                "p_A_eq": k_BA / (k_AB + k_BA),
                "p_B_eq": k_AB / (k_AB + k_BA),
            }
        else:
            if verbose:
                print("Cannot compute rates: one or both MFPTs are NaN")
            return None
    else:
        if not np.isnan(mfpt_results["mean"]):
            rate = 1.0 / mfpt_results["mean"]
            if verbose:
                print(f"Transition rate: k = {rate:.6f}")
            return {"rate": rate}
        else:
            if verbose:
                print("Cannot compute rate: MFPT is NaN")
            return None


def basinA_mfpt(x, rA=0.3):
    """Basin A for MFPT: central well around the origin."""
    return np.linalg.norm(x) < rA


def basinB_mfpt(x, r_inner=0.9, r_outer=1.2):
    """Basin B for MFPT: ring around the origin."""
    r = np.linalg.norm(x)
    return (r_inner <= r <= r_outer)

def dw1d_basin_left_mfpt(x, x_cut=0.0):
    """
    Left basin for 1D double-well in MFPT calculations:
    x < x_cut (default: negative well).
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    return x_arr[0] < x_cut


def dw1d_basin_right_mfpt(x, x_cut=0.0):
    """
    Right basin for 1D double-well in MFPT calculations:
    x > x_cut (default: positive well).
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    return x_arr[0] > x_cut


# =========================
# NEW: multi-basin MFPT
# =========================

def _multi_basin_single_passage(args):
    """Run one trajectory for multi-basin first-exit MFPT estimation.

    Starting from a random point in basin *start_id*, integrate until
    the trajectory enters a **different** basin or *max_time* elapses.

    Parameters
    ----------
    args : tuple
        Packed fields (for multiprocessing):

        0. potential, 1. basin_network, 2. start_id, 3. max_time,
        4. dt, 5. gamma, 6. kT, 7. m, 8. initial_velocity_mode,
        9. regime, 10. diffusion, 11. overdamped_eps,
        [12. bounds, 13. boundary, 14. seed]

    Returns
    -------
    tuple (start_id, target_id, t_first_hit)
        *target_id* is ``None`` if no exit occurred within *max_time*.
    """
    (
        potential,
        basin_network,
        start_id,
        max_time,
        dt,
        gamma,
        kT,
        m,
        initial_velocity_mode,
        regime,
        diffusion,
        overdamped_eps,
    ) = args[:12]

    bounds = args[12] if len(args) > 12 else None
    boundary = args[13] if len(args) > 13 else None
    seed = args[14] if len(args) > 14 else None

    if seed is not None:
        np.random.seed(int(seed) % (2**32 - 1))
        rng = np.random.RandomState(int(seed) % (2**32 - 1))
    else:
        rng = None

    # Sample initial position in starting basin
    x0 = basin_network.sample_point_in_basin(int(start_id), rng=rng)
    if x0 is None:
        return (int(start_id), None, None)

    # ------------------------------------------------------------------
    # Initial velocity
    #
    # - For overdamped dynamics it is unused, but we keep v=0 for
    #   interface consistency.
    # - Accept common string modes: "zero"/"thermal"/"boltzmann".
    # ------------------------------------------------------------------
    x = np.asarray(x0, dtype=float).ravel()
    dim = int(x.size)

    mode = initial_velocity_mode
    if str(regime).lower().startswith("over"):
        v0 = np.zeros(dim, dtype=float)
    elif mode is None:
        v0 = np.zeros(dim, dtype=float)
    elif isinstance(mode, str):
        mlow = mode.strip().lower()
        if mlow in ("zero", "zeros", "none", "null"):
            v0 = np.zeros(dim, dtype=float)
        elif mlow in ("thermal", "boltzmann", "maxwell", "maxwell-boltzmann", "mb"):
            sigma = float(np.sqrt(float(kT) / float(m)))
            # prefer per-worker RNG if available
            _rng = rng if rng is not None else np.random
            v0 = _rng.normal(0.0, sigma, dim)
        else:
            raise ValueError(f"Unknown initial_velocity_mode={mode!r}. Use 'zero' or 'thermal'.")
    else:
        v0 = np.asarray(mode, dtype=float).ravel()
        if v0.size == 1:
            v0 = np.full(dim, float(v0[0]), dtype=float)
        if v0.size != dim:
            raise ValueError(
                f"initial_velocity_mode has size {v0.size}, but position dimension is {dim}."
            )

    v = np.asarray(v0, dtype=float).ravel()
    t = 0.0

    # Enforce bounds at t=0
    if bounds is not None and boundary is not None:
        if str(boundary) == "reflect":
            if str(regime) == "overdamped":
                x = _apply_bounds(x, bounds, mode="reflect")
            else:
                x, v = _reflect_xv(x, v, bounds)
        else:
            x = _apply_bounds(x, bounds, mode=str(boundary))

    # Ensure we start in the intended basin
    current_basin = basin_network.which_basin(x)
    if current_basin != int(start_id):
        return (int(start_id), None, None)

    if str(regime) == "overdamped":
        beta = 1.0 / float(kT)
        if diffusion is None:
            diffusion = GammaToDiffusion(gamma=gamma, kT=kT)

        n_steps = int(np.floor(float(max_time) / float(dt)))
        for _ in range(n_steps):
            x = overdamped_step_euler(
                potential, x, float(dt), beta, diffusion, eps=float(overdamped_eps)
            )
            if bounds is not None and boundary is not None:
                x = _apply_bounds(x, bounds, mode=str(boundary))
            t += float(dt)
            b = basin_network.which_basin(x)
            if b is not None and int(b) != int(start_id):
                return (int(start_id), int(b), float(t))

        return (int(start_id), None, None)

    # underdamped
    n_steps = int(np.floor(float(max_time) / float(dt)))
    for _ in range(n_steps):
        x, v, _ = baobab_step(potential, x, v, float(dt), gamma, kT, m)
        if bounds is not None and boundary is not None:
            if str(boundary) == "reflect":
                x, v = _reflect_xv(x, v, bounds)
            else:
                x = _apply_bounds(x, bounds, mode=str(boundary))
        t += float(dt)
        b = basin_network.which_basin(x)
        if b is not None and int(b) != int(start_id):
            return (int(start_id), int(b), float(t))

    return (int(start_id), None, None)



# ---------------------------------------------------------------------------
# Multiprocessing helpers for MFPT network estimation
# ---------------------------------------------------------------------------
_MFPT_POOL_GLOBALS = {}

def _mfpt_pool_init(potential, basin_network, dt, max_time, D, beta, bounds, boundary):
    """Initializer for multiprocessing pool used by overdamped MFPT.

    Notes
    -----
    Internally we call the unified `_multi_basin_single_passage` helper which
    supports multiple regimes. Here we configure it for *overdamped* dynamics.
    """
    global _MFPT_POOL_GLOBALS
    # Store in a module-global dict to avoid pickling big objects per task.
    _MFPT_POOL_GLOBALS = {
        "potential": potential,
        "basin_network": basin_network,
        "dt": float(dt),
        "max_time": float(max_time),
        # Unified single-passage expects kT, not beta.
        "kT": float(1.0 / beta),
        # Overdamped config
        "gamma": 1.0,                 # unused in overdamped branch
        "m": 1.0,                     # unused in overdamped branch
        "initial_velocity_mode": "zero",  # unused in overdamped branch
        "regime": "overdamped",
        "diffusion": float(D),
        "overdamped_eps": 1e-12,
        "bounds": bounds,
        "boundary": boundary,
    }


def _multi_basin_single_passage_from_globals(args):
    """Worker wrapper (overdamped).

    Parameters
    ----------
    args : tuple
        (start_id, trial_id, seed)
    """
    start_id, trial_id, seed = args
    cfg = _MFPT_POOL_GLOBALS

    # Unified `_multi_basin_single_passage` expects:
    # (potential, basin_network, start_id, max_time, dt,
    #  gamma, kT, m, initial_velocity_mode, regime,
    #  diffusion, overdamped_eps, [bounds], [boundary], [seed])
    return _multi_basin_single_passage(
        (
            cfg["potential"],
            cfg["basin_network"],
            start_id,
            cfg["max_time"],
            cfg["dt"],
            cfg["gamma"],
            cfg["kT"],
            cfg["m"],
            cfg["initial_velocity_mode"],
            cfg["regime"],
            cfg["diffusion"],
            cfg["overdamped_eps"],
            cfg["bounds"],
            cfg["boundary"],
            seed,
        )
    )
def _multi_basin_single_passage_batch_from_globals(args):
    """Run a small batch of trajectories in a worker to reduce IPC overhead."""
    start_id, n_trials, seed0 = args
    rng = np.random.RandomState(int(seed0) % (2**31 - 1))
    out = []
    for k in range(int(n_trials)):
        seed = int(rng.randint(0, 2**31 - 1))
        out.append(_multi_basin_single_passage_from_globals((int(start_id), k, seed)))
    return out



def compute_mfpt_network(
    potential,
    basin_network,
    dt: float = 0.01,
    max_time: float = 100.0,
    D=None,
    beta: float | None = None,
    bounds=None,
    boundary: str = "reflect",
    trials_per_basin: int | None = None,
    n_procs: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    mp_start_method: str | None = None,
    **legacy_kwargs,
):
    """Estimate a multi-basin MFPT network via overdamped BD first-exit simulations.

    For each basin in *basin_network*, *trials_per_basin* overdamped
    Brownian dynamics trajectories are launched from the basin minimum.
    The first basin hit (if any) within *max_time* is recorded.  The
    returned dictionary contains MFPT matrices, exit-count tables, and
    first-exit statistics suitable for CTMC rate-matrix estimation via
    :func:`estimate_rate_matrix`.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)``.
    basin_network : BasinNetwork
        Multi-basin partition on a 2D grid.
    dt : float
        Integration time step.
    max_time : float
        Maximum simulation time per trajectory.
    D : float or ndarray or None
        Diffusion coefficient (default 1.0).
    beta : float or None
        Inverse temperature :math:`1/(k_BT)` (default 1.0).
    bounds : sequence of (lo, hi) or None
        Domain bounds.
    boundary : str
        Bound enforcement mode (default ``'reflect'``).
    trials_per_basin : int or None
        Number of trajectories per basin (default 1000).
    n_procs : int or None
        Worker processes (default 1 = serial).
    seed : int or None
        Master seed for reproducibility.
    batch_size : int or None
        Trajectories per pool task; increase (20–100) to reduce IPC
        overhead.
    mp_start_method : str or None
        Force multiprocessing context (e.g. ``'fork'``, ``'spawn'``).

    Returns
    -------
    dict
        Keys include ``'mfpt'``, ``'mfpt_matrix'``, ``'n_samples'``,
        ``'exit_to_counts'``, ``'censored_counts'``,
        ``'transition_times'``, ``'first_exit_times'``,
        ``'attempts_per_basin'``, ``'params'``, ``'method'``.

    Performance Notes
    -----------------
    When *n_procs > 1*, a multiprocessing pool is created with an
    initializer so that large objects (potential, basin_network) are
    transmitted only once per worker.  Trials are grouped into small
    batches to reduce inter-process communication overhead.
    """

    _MISSING = object()

    def _legacy_warn(msg: str):
        warnings.warn(msg, UserWarning, stacklevel=2)

    # --- legacy alias compatibility ---
    old_trials = legacy_kwargs.pop("n_trials_per_basin", _MISSING)
    if old_trials is not _MISSING:
        if trials_per_basin is not None:
            _legacy_warn(
                "Both 'trials_per_basin' and legacy 'n_trials_per_basin' were provided; "
                "using 'trials_per_basin'."
            )
        else:
            trials_per_basin = old_trials
            _legacy_warn(
                "Legacy argument 'n_trials_per_basin' is deprecated; use 'trials_per_basin'."
            )

    old_processes = legacy_kwargs.pop("processes", _MISSING)
    if old_processes is not _MISSING:
        if n_procs is not None:
            _legacy_warn(
                "Both 'n_procs' and legacy 'processes' were provided; using 'n_procs'."
            )
        else:
            n_procs = old_processes
            _legacy_warn("Legacy argument 'processes' is deprecated; use 'n_procs'.")

    old_base_seed = legacy_kwargs.pop("base_seed", _MISSING)
    if old_base_seed is not _MISSING:
        if seed is not None:
            _legacy_warn(
                "Both 'seed' and legacy 'base_seed' were provided; using 'seed'."
            )
        else:
            seed = old_base_seed
            _legacy_warn("Legacy argument 'base_seed' is deprecated; use 'seed'.")

    old_diffusion = legacy_kwargs.pop("diffusion", _MISSING)
    if old_diffusion is not _MISSING:
        if D is not None:
            _legacy_warn(
                "Both 'D' and legacy 'diffusion' were provided; using 'D'."
            )
        else:
            D = old_diffusion
            _legacy_warn("Legacy argument 'diffusion' is deprecated; use 'D'.")

    old_kT = legacy_kwargs.pop("kT", _MISSING)
    if old_kT is not _MISSING:
        if beta is not None:
            _legacy_warn(
                "Both 'beta' and legacy 'kT' were provided; using 'beta'."
            )
        else:
            kT_val = float(old_kT)
            if kT_val <= 0.0:
                raise ValueError("Legacy argument 'kT' must be > 0.")
            beta = 1.0 / kT_val
            _legacy_warn("Legacy argument 'kT' is deprecated; use 'beta'.")

    old_regime = legacy_kwargs.pop("regime", _MISSING)
    if old_regime is not _MISSING:
        _legacy_warn(
            "Legacy argument 'regime' is accepted for compatibility; "
            "compute_mfpt_network() supports only overdamped mode."
        )
        if str(old_regime).strip().lower() != "overdamped":
            raise ValueError(
                "compute_mfpt_network compatibility mode only supports regime='overdamped'."
            )

    # Accepted-but-ignored legacy knobs (function is overdamped-only).
    for legacy_name in ("gamma", "m", "initial_velocity", "verbose", "overdamped_eps"):
        if legacy_name in legacy_kwargs:
            legacy_kwargs.pop(legacy_name)
            _legacy_warn(
                f"Legacy argument '{legacy_name}' is accepted for compatibility and ignored "
                "by compute_mfpt_network()."
            )

    if legacy_kwargs:
        bad = ", ".join(sorted(str(k) for k in legacy_kwargs.keys()))
        raise TypeError(
            f"compute_mfpt_network() got unexpected keyword argument(s): {bad}"
        )

    if D is None:
        D = 1.0
    if beta is None:
        beta = 1.0
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("beta must be > 0.")

    if trials_per_basin is None:
        trials_per_basin = 1000

    n_basins = int(getattr(basin_network, "n_basins", len(basin_network.basins)))
    trials_per_basin = int(trials_per_basin)

    transition_times = [[[] for _ in range(n_basins)] for _ in range(n_basins)]
    first_exit_times = [[] for _ in range(n_basins)]
    exit_to_counts = np.zeros((n_basins, n_basins), dtype=int)
    censored_counts = np.zeros(n_basins, dtype=int)

    # Deterministic seeding for reproducible batching
    rng_master = np.random.RandomState(seed if seed is not None else None)

    # Serial fallback
    if n_procs is None:
        n_procs = 1
    n_procs = int(n_procs)
    try:
        cpu_cap = int(os.cpu_count() or 1)
    except Exception:
        cpu_cap = 1
    if n_procs < 1:
        n_procs = 1
    n_procs = min(n_procs, cpu_cap)

    # Choose a sensible default batch_size (number of trajectories per task) to
    # balance IPC overhead vs. load balancing.
    if batch_size is None:
        # Aim for ~4 tasks per worker across all basins (heuristic).
        target_total_tasks = max(1, n_procs * 4)
        tasks_per_basin = max(1, int(np.ceil(target_total_tasks / max(1, n_basins))))
        batch_size = int(np.ceil(trials_per_basin / tasks_per_basin))
    batch_size = max(1, int(batch_size))

    # Avoid spawning more workers than tasks: if we have fewer batches than
    # processes, extra workers just add start-up and IPC overhead.
    n_tasks_total = n_basins * int(np.ceil(trials_per_basin / batch_size))
    n_procs = min(n_procs, max(1, n_tasks_total))

    def _accumulate_triplet(start_id, dest_id, t):
        if dest_id is None:
            censored_counts[int(start_id)] += 1
            return
        i = int(start_id); j = int(dest_id)
        transition_times[i][j].append(float(t))
        first_exit_times[i].append(float(t))
        exit_to_counts[i, j] += 1

    if n_procs <= 1:
        # pure-python loop
        for i in range(n_basins):
            for k in range(trials_per_basin):
                s = int(rng_master.randint(0, 2**31 - 1))
                # Unified helper uses kT and an explicit regime selector.
                kT = 1.0 / beta
                start_id, dest_id, t = _multi_basin_single_passage(
                    (potential, basin_network, i, max_time, dt, 1.0, kT, 1.0, "zero", "overdamped", D, 1e-12, bounds, boundary, s)
                )
                _accumulate_triplet(start_id, dest_id, t)
    else:
        # multiprocessing with worker globals + batching
        if mp_start_method is None:
            # Prefer fork when available (Linux) to reduce pickling/copies.
            mp_start_method = "fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()
        ctx = mp.get_context(mp_start_method)

        tasks = []
        for i in range(n_basins):
            remaining = trials_per_basin
            while remaining > 0:
                n_this = min(batch_size, remaining)
                seed0 = int(rng_master.randint(0, 2**31 - 1))
                tasks.append((i, n_this, seed0))
                remaining -= n_this

        # NOTE: Avoid using the Pool context-manager here because it calls
        # terminate() on exit. terminate() can introduce large waits on some
        # platforms and makes profiling misleading (parent blocks in recv()).
        # We explicitly close() and join() once all results have been collected.
        pool = ctx.Pool(
            processes=n_procs,
            initializer=_mfpt_pool_init,
            initargs=(potential, basin_network, dt, max_time, D, beta, bounds, boundary),
        )
        try:
            # Slightly larger chunks can reduce IPC overhead for many small batches.
            chunksize = max(1, len(tasks) // max(1, n_procs * 8))
            for batch in pool.imap_unordered(
                _multi_basin_single_passage_batch_from_globals, tasks, chunksize=chunksize
            ):
                for start_id, dest_id, t in batch:
                    _accumulate_triplet(start_id, dest_id, t)
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
            pool.join()
            raise

    # Compute MFPT and sample counts
    mfpt = np.full((n_basins, n_basins), np.nan, dtype=float)
    n_samples = np.zeros((n_basins, n_basins), dtype=int)
    for i in range(n_basins):
        for j in range(n_basins):
            if i == j:
                continue
            times = transition_times[i][j]
            if len(times) > 0:
                mfpt[i, j] = float(np.mean(times))
                n_samples[i, j] = int(len(times))

    # NOTE on output schema
    # ---------------------
    # Downstream code in this project (and older user scripts) sometimes
    # expect different key names. In particular:
    #   - 'mfpt_matrix' (instead of 'mfpt')
    #   - 'success_counts' (instead of 'exit_to_counts')
    #   - 'attempts_per_basin' (instead of inferring from trials_per_basin)
    #
    # Providing these aliases makes estimate_rate_matrix(method='ctmc') and
    # other consumers robust without breaking existing callers.
    attempts_per_basin = np.full(int(n_basins), int(trials_per_basin), dtype=int)
    std_matrix = np.zeros_like(mfpt, dtype=float)

    return {
        "n_basins": int(n_basins),
        # canonical MFPT matrix
        "mfpt": mfpt,
        "mfpt_matrix": mfpt,
        "std_matrix": std_matrix,
        "n_samples": n_samples,
        # first-exit bookkeeping
        "transition_times": transition_times,
        "first_exit_times": first_exit_times,
        "exit_to_counts": exit_to_counts,
        "success_counts": exit_to_counts,
        "censored_counts": censored_counts,
        "attempts_per_basin": attempts_per_basin,
        "params": dict(
            dt=float(dt),
            max_time=float(max_time),
            D=D,
            beta=float(beta),
            bounds=bounds,
            boundary=str(boundary),
            trials_per_basin=int(trials_per_basin),
            n_procs=int(n_procs),
            batch_size=int(batch_size),
            mp_start_method=str(mp_start_method),
            seed=seed,
        ),
        "method": "traj_first_exit",
    }


def compute_mfpt_network_fpe(
    basin_network: BasinNetwork,
    D,
    beta,
    initial_distribution="boltzmann",
    sparse=True,
    verbose=True,
):
    """
    Compute MFPTs between all basins in a BasinNetwork by solving the
    discrete Fokker–Planck / backward Kolmogorov equation on the 2D grid.

    Parameters
    ----------
    basin_network : BasinNetwork
        Basin partition built on top of a 2D FES grid (xs, ys, U, labels).
    D : float or 2D array (nx, ny)
        Scalar diffusion coefficient D(x,y) on the same grid as U.
    beta : float
        Inverse temperature 1 / (k_B T). Must be consistent with the
        units of U.
    initial_distribution : {"boltzmann", "uniform"}, optional
        How to initialise within each basin when averaging MFPTs.
        "boltzmann" (default) uses exp(-beta U) restricted to the basin,
        "uniform" uses a flat distribution over grid points in the basin.
    sparse : bool, optional
        Passed through to build_fp_generator_from_fes.
    verbose : bool, optional
        If True, prints a small table of MFPTs for each target basin.

    Returns
    -------
    results : dict
        Compatible with `estimate_rate_matrix`, with keys:
        - "n_basins"
        - "mfpt_matrix"
        - "std_matrix" (zeros, placeholder)
        - "generator" (the FP generator L)
        - "beta"
        - "D"
        - "method" = "fpe_grid"
    """
    if not _HAVE_SCIPY_SPARSE_LINALG:
        raise ImportError(
            "compute_mfpt_network_fpe requires scipy.sparse.linalg. "
            "Install SciPy or use the trajectory-based compute_mfpt_network."
        )

    xs = np.asarray(basin_network.xs, dtype=float)
    ys = np.asarray(basin_network.ys, dtype=float)
    U = np.asarray(basin_network.U, dtype=float)
    labels = np.asarray(basin_network.labels, dtype=int)

    if U.shape != labels.shape:
        raise ValueError(
            "basin_network.U and basin_network.labels must have the same shape."
        )

    nx, ny = U.shape
    N = nx * ny

    U_flat = U.reshape(N)
    labels_flat = labels.reshape(N)

    n_basins = len(basin_network.basins)

    # Discrete FP generator on the grid
    L = build_fp_generator_from_fes(xs, ys, U, D, beta, sparse=sparse)

    mfpt_matrix = np.full((n_basins, n_basins), np.nan, dtype=float)

    for target_id in range(n_basins):
        # Cells belonging to the absorbing basin
        absorbing_mask = labels_flat == target_id
        if not absorbing_mask.any():
            continue  # empty basin, nothing to do

        unknown_mask = ~absorbing_mask
        unknown_idx = np.where(unknown_mask)[0]

        if unknown_idx.size == 0:
            continue

        # Restrict generator to unknown states and solve:
        #  -1 = Σ_j L_ij T_j  on unknown states
        A = L[unknown_idx, :][:, unknown_idx]
        b = -np.ones(unknown_idx.size, dtype=float)

        T_unknown = _spla.spsolve(A, b)

        # Rebuild full T field on the grid
        T_full = np.zeros(N, dtype=float)
        T_full[absorbing_mask] = 0.0
        T_full[unknown_idx] = T_unknown

        # Average over initial distributions in each source basin
        for source_id in range(n_basins):
            if source_id == target_id:
                continue

            src_mask = labels_flat == source_id
            if not src_mask.any():
                continue  # empty basin, skip

            if initial_distribution == "boltzmann":
                # Numerically stable Boltzmann weights: shift by the minimum
                # energy within the source basin so the exponent is <= 0.
                u = U_flat[src_mask]
                u0 = float(np.min(u))
                w = np.exp(-float(beta) * (u - u0))
                if not np.all(np.isfinite(w)) or w.sum() <= 0.0:
                    # Fall back to uniform if e^{-β(U-U0)} under/overflows
                    w = np.ones(src_mask.sum(), dtype=float)
            elif initial_distribution == "uniform":
                w = np.ones(src_mask.sum(), dtype=float)
            else:
                raise ValueError(
                    "initial_distribution must be 'boltzmann' or 'uniform'."
                )

            w /= w.sum()
            tau = float((w * T_full[src_mask]).sum())
            mfpt_matrix[source_id, target_id] = tau

        if verbose:
            row = ", ".join(
                f"{mfpt_matrix[s, target_id]:.3g}"
                if np.isfinite(mfpt_matrix[s, target_id])
                else "nan"
                for s in range(n_basins)
            )
            print(f"[FPE MFPT] target basin {target_id}: {row}")

    std_matrix = np.zeros_like(mfpt_matrix)

    return {
        "n_basins": n_basins,
        "mfpt_matrix": mfpt_matrix,
        "std_matrix": std_matrix,
        "generator": L,
        "beta": beta,
        "D": D,
        "method": "fpe_grid",
    }


def estimate_rate_matrix(mfpt_network_results, verbose=True, method="auto"):
    """
    Estimate a continuous-time Markov chain (CTMC) generator K between basins.

    Preferred (and default) behavior:
      - If first-exit CTMC statistics are available (from trajectory MFPT-network runs),
        return the consistent generator constructed as:

            k_out(i) = 1 / E[t_exit | exit]
            p_ij     = P(next basin = j | exit from i)
            K_ij     = k_out(i) * p_ij  (j != i)
            K_ii     = -sum_{j != i} K_ij

        This is the proper CTMC model implied by *first-exit* trajectories.

    Fallback:
      - If those stats are not present (or method="inverse_mfpt"), estimate
        pairwise rates as k_ij ≈ 1 / MFPT(i->j). This is generally NOT a
        consistent generator for multi-basin systems, but is kept for backward
        compatibility.

    Parameters
    ----------
    mfpt_network_results : dict
        Output of compute_mfpt_network / compute_mfpt_network_fpe / etc.
    verbose : bool
        Print diagnostics.
    method : {"auto", "ctmc", "inverse_mfpt"}
        Selection of estimator.

    Returns
    -------
    K : (n_basins, n_basins) ndarray
        CTMC generator (rows sum to zero).
    """
    if method is None:
        method = "auto"
    method = str(method).lower()

    # Be tolerant to older/newer schemas.
    if "n_basins" in mfpt_network_results:
        n = int(mfpt_network_results["n_basins"])
    else:
        mat = None
        for k in ("mfpt_matrix", "mfpt", "mfpt_mat", "MFPT"):
            if k in mfpt_network_results:
                mat = np.asarray(mfpt_network_results[k])
                break
        if mat is None or mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise KeyError(
                "Could not infer number of basins: provide 'n_basins' or an MFPT matrix under one of "
                "['mfpt_matrix','mfpt','mfpt_mat','MFPT']."
            )
        n = int(mat.shape[0])

    def _ctmc_from_first_exit(res):
        # If the generator was precomputed, use it.
        if "ctmc_generator" in res and res["ctmc_generator"] is not None:
            K = np.asarray(res["ctmc_generator"], dtype=float)
            if K.shape != (n, n):
                raise ValueError("ctmc_generator has wrong shape.")
            return K

        # Otherwise, compute from first-exit data.
        # Accept multiple schema variants.
        if "success_counts" not in res and "exit_to_counts" in res:
            res["success_counts"] = res["exit_to_counts"]
        if "attempts_per_basin" not in res:
            # Common source: scalar trials_per_basin stored under params
            trials = None
            if "params" in res and isinstance(res["params"], dict):
                trials = res["params"].get("trials_per_basin", None)
            if trials is None:
                trials = res.get("trials_per_basin", None)
            if trials is not None:
                res["attempts_per_basin"] = np.full(n, int(trials), dtype=int)

        if "success_counts" not in res or "transition_times" not in res or "attempts_per_basin" not in res:
            raise KeyError(
                "Missing first-exit fields. Need (success_counts or exit_to_counts), transition_times, "
                "and (attempts_per_basin or trials_per_basin)."
            )

        success_counts = np.asarray(res["success_counts"], dtype=int)
        attempts_per_basin = np.asarray(res["attempts_per_basin"], dtype=int)
        if attempts_per_basin.ndim == 0:
            attempts_per_basin = np.full(n, int(attempts_per_basin), dtype=int)
        transition_times = res["transition_times"]

        # Exit times per basin (successful exits only)
        exit_times = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                exit_times[i].extend(transition_times[i][j])

        exit_mean = np.full(n, np.nan, dtype=float)
        exit_counts = np.zeros(n, dtype=int)
        for i in range(n):
            arr = np.asarray(exit_times[i], dtype=float)
            exit_counts[i] = int(arr.size)
            if arr.size > 0:
                exit_mean[i] = float(arr.mean())

        K = np.zeros((n, n), dtype=float)
        for i in range(n):
            if exit_counts[i] > 0 and np.isfinite(exit_mean[i]) and exit_mean[i] > 0.0:
                k_out = 1.0 / float(exit_mean[i])
                denom = float(exit_counts[i])
                for j in range(n):
                    if i == j:
                        continue
                    c = int(success_counts[i, j])
                    if c > 0:
                        K[i, j] = k_out * (c / denom)
                K[i, i] = -float(np.sum(K[i, :]))
            else:
                K[i, i] = 0.0

        if verbose:
            censored = attempts_per_basin - exit_counts
            print("[Rate matrix] CTMC generator from first-exit statistics:")
            print("  (rows sum to ~0; diag = -sum off-diag)")
            print(K)
            print("[Rate matrix] First-exit diagnostics:")
            for i in range(n):
                print(f"  basin {i}: exits={exit_counts[i]}/{attempts_per_basin[i]}  censored={censored[i]}  <t_exit>={exit_mean[i]}")
        return K

    if method in ("auto", "ctmc"):
        try:
            return _ctmc_from_first_exit(mfpt_network_results)
        except Exception as e:
            if method == "ctmc":
                raise
            if verbose:
                print(f"[Rate matrix] CTMC estimator unavailable ({type(e).__name__}: {e}); falling back to inverse MFPT.")

    # Fallback: inverse MFPT (pairwise)
    # Fallback: inverse MFPT (pairwise). Accept a few common key variants.
    mfpt_src = mfpt_network_results.get("mfpt_matrix", None)
    if mfpt_src is None:
        mfpt_src = mfpt_network_results.get("mfpt", mfpt_network_results.get("mfpt_mat", None))
    if mfpt_src is None:
        raise KeyError("Missing MFPT matrix field (expected 'mfpt_matrix' or 'mfpt').")
    mfpt = np.asarray(mfpt_src, dtype=float)
    if mfpt.shape != (n, n):
        raise ValueError("mfpt_matrix has wrong shape.")

    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tau = mfpt[i, j]
            if np.isfinite(tau) and tau > 0.0:
                K[i, j] = 1.0 / tau
        K[i, i] = -float(np.sum(K[i, :]))

    if verbose:
        print("[Rate matrix] Inverse-MFPT approximation k_ij ≈ 1/MFPT(i->j):")
        print(K)

    return K
