"""stochkin.replicas
==================

Parallel replica dynamics for sampling free-energy landscapes.

This module runs *M* independent short simulations (“replicas”) in
parallel and accumulates position- and energy-histograms.  The
averaged histograms approximate the canonical (Boltzmann) distribution
and can be compared with the analytic free-energy surface.

- :func:`run_replicas` / :func:`single_replica` – 2D replicas
  (underdamped Langevin via BAOAB or overdamped BD).
- :func:`run_replicas_1d` / :func:`single_replica_1d` – 1D replicas.

All replica functions are multiprocessing-safe (top-level, picklable).
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from .integrators import baobab_2d, overdamped_bd


# ======================================================================
# Single 2D replica
# ======================================================================

def single_replica(args):
    """Run a single 2D replica and return position/energy histograms.

    This is a top-level, picklable worker for ``multiprocessing.Pool``.

    Parameters
    ----------
    args : tuple
        Packed fields:

        0. potential, 1. max_time, 2. dt, 3. gamma, 4. kT,
        5. initial_position, 6. initial_velocity,
        7. save_frequency, 8. bins2d, 9. binsE, 10. m,
        11. pos_range, 12. energy_range,
        13. regime (``'underdamped'`` or ``'overdamped'``),
        14. diffusion,
        15. burn_in_steps, 16. bounds, 17. boundary, 18. seed.

    Returns
    -------
    Hpos : ndarray, shape (nx, ny)
        2D position histogram.
    HE : ndarray, shape (binsE,)
        1D energy histogram.
    """
    (
        potential, max_time, dt, gamma, kT,
        initial_position, initial_velocity,
        save_frequency, bins2d, binsE, m,
        pos_range, energy_range,
        regime, diffusion,
        burn_in_steps, bounds, boundary, seed,
    ) = args

    if seed is not None:
        np.random.seed(int(seed) % (2**32 - 1))

    if regime == "overdamped":
        times, positions, velocities, energies = overdamped_bd(
            potential=potential,
            max_time=max_time,
            dt=dt,
            kT=kT,
            initial_position=initial_position,
            diffusion=diffusion,
            gamma=gamma,  # used only if diffusion is None
            save_frequency=save_frequency,
            burn_in_steps=burn_in_steps,
            bounds=bounds,
            boundary=boundary,
            seed=seed,
        )
    else:
        times, positions, velocities, energies = baobab_2d(
            potential,
            max_time=max_time,
            dt=dt,
            gamma=gamma,
            kT=kT,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            save_frequency=save_frequency,
            m=m,
        )
        # Apply burn-in by discarding samples before t0
        if burn_in_steps and burn_in_steps > 0 and times.size > 0:
            t0 = float(burn_in_steps) * float(dt)
            keep = times >= t0
            positions = positions[keep]
            energies = energies[keep]

    if positions.size == 0:
        Hpos = np.zeros(
            (bins2d[0], bins2d[1]) if isinstance(bins2d, (tuple, list)) else (int(bins2d), int(bins2d)),
            dtype=float,
        )
        HE = np.zeros(int(binsE), dtype=float)
        return Hpos, HE

    Hpos, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=bins2d,
        range=pos_range,
    )

    HE, _ = np.histogram(
        energies,
        bins=binsE,
        range=energy_range,
    )
    return Hpos, HE


# ======================================================================
# Run 2D replicas
# ======================================================================

def run_replicas(
    potential,
    M=10,
    max_time=200,
    dt=0.01,
    gamma=10.0,
    kT=0.05,
    initial_position=(0.0, -0.70),
    initial_velocity=(0.0, 0.05),
    save_frequency=10,
    bins=100,
    m=1.0,
    x_range=((-2.0, 2.0), (-2.0, 2.0)),
    energy_range=(-2.0, 5.0),
    regime="underdamped",
    diffusion=None,
    processes=None,
    # New (backwards-compatible keywords)
    bins_energy=None,
    burn_in_fraction=0.0,
    bounds=None,
    boundary="reflect",
    base_seed=None,
    plot=True,
):
    """Run *M* independent 2D replicas in parallel and average histograms.

    Each replica is a short simulation (underdamped Langevin or
    overdamped BD) with slightly perturbed initial conditions.
    Histograms of position and energy are accumulated and averaged.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)``.
    M : int
        Number of replicas.
    max_time : float
        Integration time per replica.
    dt : float
        Time step.
    gamma : float
        Friction coefficient.
    kT : float
        Thermal energy.
    initial_position : array_like
        Nominal starting position (slightly perturbed per replica).
    initial_velocity : array_like
        Nominal starting velocity.
    save_frequency : int
        Steps between saved frames.
    bins : int or (int, int)
        Histogram bins for position (``nx`` or ``(nx, ny)``).
    m : float
        Mass.
    x_range : ((xmin, xmax), (ymin, ymax))
        Position histogram range.
    energy_range : (emin, emax)
        Energy histogram range.
    regime : {'underdamped', 'overdamped'}
        Dynamics type.
    diffusion : scalar, callable, or None
        Diffusion for overdamped mode.
    processes : int or None
        Worker processes (None or 1 = serial).
    bins_energy : int or None
        Separate energy-histogram bin count (defaults to *bins*).
    burn_in_fraction : float
        Fraction of initial trajectory to discard before histogramming.
    bounds : sequence of (lo, hi) or None
        Domain bounds (overdamped only).
    boundary : str
        Bound enforcement mode (default ``'reflect'``).
    base_seed : int or None
        Master seed for reproducibility.
    plot : bool
        If ``True`` (default), show diagnostic position and energy plots.

    Returns
    -------
    hist2d_avg : ndarray
        Averaged 2D position histogram.
    histE_avg : ndarray
        Averaged 1D energy histogram.
    """
    pos_range = x_range

    # 2D histogram bins
    if isinstance(bins, (tuple, list, np.ndarray)):
        if len(bins) != 2:
            raise ValueError("bins tuple must be (nx, ny).")
        bins2d = (int(bins[0]), int(bins[1]))
        binsE = int(bins_energy) if bins_energy is not None else 100
    else:
        bins2d = int(bins)
        binsE = int(bins_energy) if bins_energy is not None else int(bins)

    # Burn-in in *integration steps*
    n_steps = int(np.floor(float(max_time) / float(dt)))
    burn_in_steps = int(np.floor(float(burn_in_fraction) * n_steps))

    rng = np.random.RandomState(int(base_seed)) if base_seed is not None else np.random

    # Prepare argument list — each replica gets independent initial conditions + seed
    args = []
    for r in range(int(M)):
        init_pos = np.array(initial_position, dtype=float).ravel()
        init_vel = np.array(initial_velocity, dtype=float).ravel()

        init_pos = init_pos + 0.01 * rng.randn(init_pos.size)
        init_vel = init_vel + 0.01 * rng.randn(init_vel.size)

        seed = int(rng.randint(0, 2**31 - 1))

        args.append((
            potential, max_time, dt, gamma, kT,
            init_pos, init_vel,
            save_frequency, bins2d, binsE, m,
            pos_range, energy_range,
            regime, diffusion,
            burn_in_steps, bounds, boundary, seed,
        ))

    # Run replicas
    if processes is None or int(processes) == 1:
        results = [single_replica(a) for a in args]
    else:
        pool = Pool(processes=int(processes))
        try:
            chunksize = max(1, len(args) // (int(processes) * 8))
            results = list(pool.imap_unordered(single_replica, args, chunksize=chunksize))
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
            pool.join()
            raise

    # Accumulate
    hist2d_accum = np.zeros((bins2d[0], bins2d[1]) if isinstance(bins2d, tuple) else (int(bins2d), int(bins2d)))
    histE_accum = np.zeros(int(binsE))
    for Hpos, HE in results:
        hist2d_accum += Hpos
        histE_accum += HE

    hist2d_avg = hist2d_accum / float(M)
    histE_avg = histE_accum / float(M)

    if plot:
        # Position distribution
        plt.figure(figsize=(6, 5))
        plt.imshow(
            hist2d_avg.T,
            origin="lower",
            extent=[
                pos_range[0][0], pos_range[0][1],
                pos_range[1][0], pos_range[1][1],
            ],
            aspect="auto",
            cmap="viridis",
        )
        plt.colorbar(label="Average counts")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Averaged Position Distribution over {M} replicas")
        plt.tight_layout()
        plt.show()

        # Energy distribution
        plt.figure(figsize=(6, 4))
        centers = np.linspace(energy_range[0], energy_range[1], binsE)
        plt.bar(
            centers,
            histE_avg,
            width=(centers[1] - centers[0]) if centers.size > 1 else 1.0,
            alpha=0.7,
        )
        plt.xlabel("Energy")
        plt.ylabel("Average Frequency")
        plt.title(f"Averaged Energy Distribution over {M} replicas")
        plt.tight_layout()
        plt.show()

    return hist2d_avg, histE_avg


# ======================================================================
# Single 1D replica
# ======================================================================

def single_replica_1d(args):
    """Run a single 1D replica and return position/energy histograms.

    This is a top-level, picklable worker for ``multiprocessing.Pool``.

    Parameters
    ----------
    args : tuple
        Packed fields:

        0. potential, 1. max_time, 2. dt, 3. gamma, 4. kT,
        5. initial_position, 6. initial_velocity,
        7. save_frequency, 8. bins, 9. m,
        10. pos_range, 11. energy_range,
        12. regime (``'underdamped'`` or ``'overdamped'``),
        13. diffusion.

    Returns
    -------
    Hx : ndarray, shape (bins,)
        1D position histogram.
    HE : ndarray, shape (bins,)
        1D energy histogram.
    """
    (
        potential, max_time, dt, gamma, kT,
        initial_position, initial_velocity,
        save_frequency, bins, m,
        pos_range, energy_range,
        regime, diffusion
    ) = args

    if regime == "overdamped":
        times, positions, velocities, energies = overdamped_bd(
            potential=potential,
            max_time=max_time,
            dt=dt,
            kT=kT,
            initial_position=initial_position,
            diffusion=diffusion,
            gamma=gamma,
            save_frequency=save_frequency,
        )
    else:
        times, positions, velocities, energies = baobab_2d(
            potential,
            max_time=max_time,
            dt=dt,
            gamma=gamma,
            kT=kT,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            save_frequency=save_frequency,
            m=m,
        )

    x = positions[:, 0]

    Hx, _ = np.histogram(
        x,
        bins=bins,
        range=pos_range,   # FIXED: was x_range
    )

    HE, _ = np.histogram(
        energies,
        bins=bins,
        range=energy_range,
    )

    return Hx, HE


# ======================================================================
# Run 1D replicas
# ======================================================================

def run_replicas_1d(
    potential,
    M=10,
    max_time=200,
    dt=0.01,
    gamma=10.0,
    kT=0.05,
    initial_position=(0.0,),
    initial_velocity=(0.0,),
    save_frequency=10,
    bins=100,
    m=1.0,
    x_range=(-2.0, 2.0),
    energy_range=(-2.0, 5.0),
    plot=True,
    regime="underdamped",
    diffusion=None,
    processes=None,
):
    """Run *M* independent 1D replicas in parallel and average histograms.

    1D analogue of :func:`run_replicas`.  Each replica is slightly
    perturbed in initial conditions.  Position and energy histograms
    are averaged over all replicas.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)``.
    M : int
        Number of replicas.
    max_time : float
        Integration time per replica.
    dt : float
        Time step.
    gamma : float
        Friction.
    kT : float
        Thermal energy.
    initial_position : array_like
        Nominal starting position.
    initial_velocity : array_like
        Nominal starting velocity.
    save_frequency : int
        Steps between saved frames.
    bins : int
        Number of histogram bins.
    m : float
        Mass.
    x_range : (xmin, xmax)
        Position histogram range.
    energy_range : (emin, emax)
        Energy histogram range.
    plot : bool
        Show diagnostic plots.
    regime : {'underdamped', 'overdamped'}
        Dynamics type.
    diffusion : scalar, callable, or None
        Diffusion for overdamped mode.
    processes : int or None
        Worker processes (None = all CPUs).

    Returns
    -------
    Hx_avg : ndarray
        Averaged 1D position histogram.
    HE_avg : ndarray
        Averaged 1D energy histogram.
    """
    pos_range = x_range

    args = []
    for _ in range(M):
        if isinstance(initial_position, (tuple, list, np.ndarray)):
            init_pos = np.array(initial_position, dtype=float) + 0.01 * np.random.randn(len(initial_position))
        else:
            init_pos = np.array([initial_position]) + 0.01 * np.random.randn(1)

        if isinstance(initial_velocity, (tuple, list, np.ndarray)):
            init_vel = np.array(initial_velocity, dtype=float) + 0.01 * np.random.randn(len(initial_velocity))
        else:
            init_vel = np.array([initial_velocity]) + 0.01 * np.random.randn(1)

        args.append((
            potential,
            max_time,
            dt,
            gamma,
            kT,
            init_pos,
            init_vel,
            save_frequency,
            bins,
            m,
            pos_range,
            energy_range,
            regime,
            diffusion,
        ))

    with Pool(processes=processes) as pool:
        results = pool.map(single_replica_1d, args)

    Hx_list, HE_list = zip(*results)
    Hx_avg = np.mean(Hx_list, axis=0)
    HE_avg = np.mean(HE_list, axis=0)

    # --- Plot ---
    if plot:
        # Position histogram
        x_centers = np.linspace(pos_range[0], pos_range[1], bins)
        plt.figure(figsize=(6, 4))
        plt.bar(
            x_centers,
            Hx_avg,
            width=(x_centers[1] - x_centers[0]),
            alpha=0.7,
        )
        plt.xlabel("x")
        plt.ylabel("Average counts")
        plt.title(f"Averaged position distribution over {M} replicas")

        # Energy histogram
        e_centers = np.linspace(energy_range[0], energy_range[1], bins)
        plt.figure(figsize=(6, 4))
        plt.bar(
            e_centers,
            HE_avg,
            width=(e_centers[1] - e_centers[0]),
            alpha=0.7,
        )
        plt.xlabel("Energy")
        plt.ylabel("Average counts")
        plt.title(f"Averaged energy distribution over {M} replicas")

        plt.tight_layout()
        plt.show()

    return Hx_avg, HE_avg
