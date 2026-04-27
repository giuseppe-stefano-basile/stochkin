"""Fast 1D Langevin first-exit helpers.

This module keeps the accelerated backend separate from :mod:`stochkin.mfpt`
so the public MFPT API can choose it opportunistically while retaining the
existing pure-Python implementation as the portability fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .potentials import BasinNetwork1D

try:  # pragma: no cover - exercised when the extension is built in-place
    from . import _fast_langevin1d as _compiled_1d
except Exception:  # pragma: no cover - import must stay optional
    _compiled_1d = None


class FastLangevinUnsupported(RuntimeError):
    """Raised when a problem is not eligible for the fast 1D engine."""


@dataclass
class _Fast1DInputs:
    x_grid: np.ndarray
    force_grid: np.ndarray
    D_grid: np.ndarray
    gradD_grid: np.ndarray
    labels: np.ndarray
    bounds: tuple[float, float]
    boundary_mode: int


def fast_langevin1d_backend_available() -> bool:
    """Return ``True`` when the compiled 1D first-exit backend is importable."""
    return _compiled_1d is not None


def _edge_order(n: int) -> int:
    return 2 if int(n) >= 3 else 1


def _as_1d_bounds(bounds) -> tuple[float, float]:
    if bounds is None:
        raise FastLangevinUnsupported("fast 1D engine requires explicit 1D bounds")

    arr = np.asarray(bounds, dtype=float)
    if arr.shape == (2,):
        lo, hi = float(arr[0]), float(arr[1])
    elif arr.shape == (1, 2):
        lo, hi = float(arr[0, 0]), float(arr[0, 1])
    else:
        raise FastLangevinUnsupported(
            "fast 1D engine expects bounds as (lo, hi) or ((lo, hi),)"
        )

    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise FastLangevinUnsupported("fast 1D engine requires finite bounds with lo < hi")
    return lo, hi


def _boundary_mode(boundary: str) -> int:
    mode = str(boundary or "reflect").strip().lower()
    if mode == "reflect":
        return 0
    if mode == "clip":
        return 1
    raise FastLangevinUnsupported("fast 1D engine supports boundary='reflect' or 'clip'")


def _extract_grid_potential_1d(potential, basin_network: BasinNetwork1D):
    x_grid = getattr(potential, "x_grid", None)
    U_grid = getattr(potential, "fes_grid", None)
    if x_grid is None or U_grid is None:
        raise FastLangevinUnsupported(
            "fast 1D engine requires a gridded FESPotential1D-like potential"
        )

    x_grid = np.asarray(x_grid, dtype=float).ravel()
    U_grid = np.asarray(U_grid, dtype=float).ravel()
    bn_s = np.asarray(basin_network.s, dtype=float).ravel()
    labels = np.asarray(basin_network.labels, dtype=np.int64).ravel()

    if x_grid.ndim != 1 or U_grid.shape != x_grid.shape:
        raise FastLangevinUnsupported("potential x_grid and fes_grid must be 1D arrays")
    if x_grid.size < 2:
        raise FastLangevinUnsupported("fast 1D engine requires at least two grid points")
    if np.any(np.diff(x_grid) <= 0.0):
        raise FastLangevinUnsupported("potential x_grid must be strictly increasing")
    if not np.all(np.isfinite(x_grid)) or not np.all(np.isfinite(U_grid)):
        raise FastLangevinUnsupported("fast 1D engine requires finite potential grids")
    if labels.shape != x_grid.shape or bn_s.shape != x_grid.shape:
        raise FastLangevinUnsupported("basin labels must match the 1D potential grid")
    if not np.allclose(bn_s, x_grid, rtol=1e-10, atol=1e-12):
        raise FastLangevinUnsupported("basin network grid must match potential grid")

    force_grid = -np.gradient(U_grid, x_grid, edge_order=_edge_order(x_grid.size))
    if not np.all(np.isfinite(force_grid)):
        raise FastLangevinUnsupported("computed force grid contains non-finite values")

    return x_grid, force_grid.astype(float, copy=False), labels


def _prepare_diffusion_grid(D, x_grid: np.ndarray):
    if D is None:
        D = 1.0

    if np.isscalar(D):
        value = float(D)
        if not np.isfinite(value) or value < 0.0:
            raise FastLangevinUnsupported("scalar D must be finite and non-negative")
        return np.full_like(x_grid, value), np.zeros_like(x_grid)

    D_grid = np.asarray(D, dtype=float).ravel()
    if D_grid.shape != x_grid.shape:
        raise FastLangevinUnsupported(
            "fast 1D engine accepts D as a scalar or an array matching x_grid"
        )
    if not np.all(np.isfinite(D_grid)) or np.any(D_grid < 0.0):
        raise FastLangevinUnsupported("D grid must contain finite non-negative values")

    gradD_grid = np.gradient(D_grid, x_grid, edge_order=_edge_order(x_grid.size))
    if not np.all(np.isfinite(gradD_grid)):
        raise FastLangevinUnsupported("computed grad(D) grid contains non-finite values")
    return D_grid.astype(float, copy=False), gradD_grid.astype(float, copy=False)


def _prepare_fast_1d_inputs(potential, basin_network, D, bounds, boundary) -> _Fast1DInputs:
    if not isinstance(basin_network, BasinNetwork1D):
        raise FastLangevinUnsupported("fast engine currently supports BasinNetwork1D only")

    x_grid, force_grid, labels = _extract_grid_potential_1d(potential, basin_network)
    D_grid, gradD_grid = _prepare_diffusion_grid(D, x_grid)
    lo, hi = _as_1d_bounds(bounds)

    return _Fast1DInputs(
        x_grid=np.ascontiguousarray(x_grid, dtype=float),
        force_grid=np.ascontiguousarray(force_grid, dtype=float),
        D_grid=np.ascontiguousarray(D_grid, dtype=float),
        gradD_grid=np.ascontiguousarray(gradD_grid, dtype=float),
        labels=np.ascontiguousarray(labels, dtype=np.int64),
        bounds=(lo, hi),
        boundary_mode=_boundary_mode(boundary),
    )


def _seed_value(seed) -> int:
    if seed is not None:
        return int(seed) & ((1 << 64) - 1)
    state = np.random.SeedSequence().generate_state(2, dtype=np.uint64)
    return int(state[0] ^ state[1]) & ((1 << 64) - 1)


def _assemble_result(
    *,
    target_ids: np.ndarray,
    times: np.ndarray,
    n_basins: int,
    trials_per_basin: int,
    params: dict,
):
    transition_times = [[[] for _ in range(n_basins)] for _ in range(n_basins)]
    first_exit_times = [[] for _ in range(n_basins)]
    exit_to_counts = np.zeros((n_basins, n_basins), dtype=int)
    censored_counts = np.zeros(n_basins, dtype=int)

    target_ids = np.asarray(target_ids, dtype=np.int64).ravel()
    times = np.asarray(times, dtype=float).ravel()
    expected = int(n_basins) * int(trials_per_basin)
    if target_ids.size != expected or times.size != expected:
        raise RuntimeError("compiled fast engine returned arrays with unexpected length")

    for idx, dest_id in enumerate(target_ids):
        start_id = idx // int(trials_per_basin)
        if int(dest_id) < 0:
            censored_counts[start_id] += 1
            continue
        i = int(start_id)
        j = int(dest_id)
        t = float(times[idx])
        transition_times[i][j].append(t)
        first_exit_times[i].append(t)
        exit_to_counts[i, j] += 1

    mfpt = np.full((n_basins, n_basins), np.nan, dtype=float)
    n_samples = np.zeros((n_basins, n_basins), dtype=int)
    for i in range(n_basins):
        for j in range(n_basins):
            if i == j:
                continue
            samples = transition_times[i][j]
            if samples:
                mfpt[i, j] = float(np.mean(samples))
                n_samples[i, j] = int(len(samples))

    attempts_per_basin = np.full(n_basins, int(trials_per_basin), dtype=int)
    std_matrix = np.zeros_like(mfpt, dtype=float)

    return {
        "n_basins": int(n_basins),
        "mfpt": mfpt,
        "mfpt_matrix": mfpt,
        "std_matrix": std_matrix,
        "n_samples": n_samples,
        "transition_times": transition_times,
        "first_exit_times": first_exit_times,
        "exit_to_counts": exit_to_counts,
        "success_counts": exit_to_counts,
        "censored_counts": censored_counts,
        "attempts_per_basin": attempts_per_basin,
        "params": params,
        "method": "traj_first_exit_fast_1d",
    }


def compute_mfpt_network_fast_1d(
    *,
    potential,
    basin_network,
    dt: float,
    max_time: float,
    D,
    beta: float,
    bounds,
    boundary: str,
    trials_per_basin: int,
    n_threads: int | None,
    seed: int | None,
    batch_size: int | None,
    engine_requested: str,
):
    """Compute a 1D first-exit MFPT network with the compiled backend."""
    if _compiled_1d is None:
        raise FastLangevinUnsupported(
            "compiled extension stochkin._fast_langevin1d is not built"
        )

    inputs = _prepare_fast_1d_inputs(potential, basin_network, D, bounds, boundary)
    n_basins = int(getattr(basin_network, "n_basins", len(basin_network.basins)))
    trials_per_basin = int(trials_per_basin)
    if n_basins <= 0:
        raise ValueError("basin_network must contain at least one basin")
    if trials_per_basin <= 0:
        raise ValueError("trials_per_basin must be > 0")

    n_threads = int(n_threads if n_threads is not None else 1)
    if n_threads < 1:
        n_threads = 1
    n_threads = min(n_threads, int(os.cpu_count() or 1))

    seed_value = _seed_value(seed)
    lo, hi = inputs.bounds

    target_ids, times = _compiled_1d.simulate_first_exit_network(
        inputs.x_grid,
        inputs.force_grid,
        inputs.D_grid,
        inputs.gradD_grid,
        inputs.labels,
        n_basins,
        float(dt),
        float(max_time),
        float(beta),
        float(lo),
        float(hi),
        int(inputs.boundary_mode),
        int(trials_per_basin),
        int(seed_value),
        int(n_threads),
    )

    return _assemble_result(
        target_ids=target_ids,
        times=times,
        n_basins=n_basins,
        trials_per_basin=trials_per_basin,
        params=dict(
            dt=float(dt),
            max_time=float(max_time),
            D=D,
            beta=float(beta),
            bounds=bounds,
            boundary=str(boundary),
            trials_per_basin=int(trials_per_basin),
            n_procs=int(n_threads),
            batch_size=None if batch_size is None else int(batch_size),
            mp_start_method=None,
            seed=seed,
            engine_requested=str(engine_requested),
            engine_used="fast_1d_compiled",
            fast_backend="compiled",
            fast_seed=int(seed_value),
        ),
    )
