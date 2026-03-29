"""stochkin.workflows
===================

High-level, one-call wrappers for the most common stochastic-kinetics
pipelines.  Each function bundles loading, basin detection, and CTMC
generator construction into a single call and returns a uniform result
dictionary.

Result dictionary keys
----------------------
s            1-D grid (CV values)
F            Free energy at grid points  [kJ/mol]
D_used       Diffusion coefficient used on the grid  [CV²/time_unit]
kT           Thermal energy at the requested temperature  [kJ/mol]
K            Rate matrix in the input *time_unit*   [1/time_unit]
K_ps         Rate matrix in picoseconds             [1/ps]
exit_mean    Mean exit time per basin               [time_unit]
exit_ps      Mean exit time per basin               [ps]
k_out        Total exit rate per basin              [1/time_unit]
k_out_ps     Total exit rate per basin              [1/ps]
p_branch     Branching-probability matrix           [dimensionless]
labels_full  Integer label per grid point (basin id, or -1)
basin_ids    Sorted array of integer basin ids
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from itertools import combinations

from .fes import load_plumed_fes_2d, make_fes_potential_from_grid
from .potentials import build_basin_network_from_fes_1d, build_basin_network_from_potential
from .fpe import compute_ctmc_generator_fpe_1d
from .mfep import compute_mfep_profile_1d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_K_B_KJ: float = 0.008314462618  # kJ mol⁻¹ K⁻¹


# ---------------------------------------------------------------------------
# Time-unit helpers
# ---------------------------------------------------------------------------
def _time_unit_to_ps(unit: str) -> float:
    """Return the number of picoseconds in one *unit*."""
    table = {"fs": 1e-3, "ps": 1.0, "ns": 1e3, "us": 1e6, "ms": 1e9}
    key = unit.lower()
    if key not in table:
        raise ValueError(
            f"Unknown time unit '{unit}'. Supported: {list(table)}"
        )
    return table[key]


def _kT(T: float) -> float:
    return _K_B_KJ * float(T)


# ---------------------------------------------------------------------------
# Basin-core helpers
# ---------------------------------------------------------------------------
def _build_core_labels(
    s: np.ndarray,
    F: np.ndarray,
    basin_network,
    core_fraction: Optional[float],
) -> None:
    """Attach ``core_labels`` to *basin_network* in-place (modifies in place).

    Each core is the contiguous segment around the basin minimum that
    contains the closest ``core_fraction`` fraction of basin points.
    If *core_fraction* is ``None`` the full-basin labels are used.
    """
    if core_fraction is None:
        return

    labels_full = basin_network.labels
    core_labels = np.full_like(labels_full, -1, dtype=int)

    for b in basin_network.basins:
        bid = int(b.id)
        idx_b = np.flatnonzero(labels_full == bid)
        if idx_b.size == 0:
            continue

        # Locate the minimum inside the basin
        i_min = int(idx_b[np.argmin(F[idx_b])])
        mpos = float(s[i_min])

        # Sort basin points by distance to minimum; keep top fraction
        dist = np.abs(s[idx_b] - mpos)
        n_core = max(1, int(np.ceil(core_fraction * idx_b.size)))
        order = np.argsort(dist)
        core_idx = idx_b[np.sort(order[:n_core])]

        # Find the contiguous segment containing i_min
        tmp = np.zeros(s.size, dtype=bool)
        tmp[core_idx] = True
        a = i_min
        b_end = i_min
        while a - 1 >= 0 and tmp[a - 1]:
            a -= 1
        while b_end + 1 < s.size and tmp[b_end + 1]:
            b_end += 1
        core_labels[a : b_end + 1] = bid

    basin_network.core_labels = core_labels


# ---------------------------------------------------------------------------
# Result packaging
# ---------------------------------------------------------------------------
def _pack_result(
    s: np.ndarray,
    F: np.ndarray,
    D_used,
    kT_val: float,
    time_unit_ps: float,
    K_raw: np.ndarray,
    info: dict,
) -> dict:
    """Collect CTMC outputs into the canonical result dict."""
    K_raw = np.asarray(K_raw, dtype=float)
    n = K_raw.shape[0]

    K_ps = K_raw / time_unit_ps

    exit_mean = np.asarray(
        info.get("exit_mean", np.full(n, np.nan)), dtype=float
    )
    k_out = np.asarray(
        info.get("k_out", np.full(n, np.nan)), dtype=float
    )
    p_branch = np.asarray(
        info.get("p_branch", np.full((n, n), np.nan)), dtype=float
    )
    labels_full = np.asarray(
        info.get("labels_full", np.zeros(s.size, dtype=int)), dtype=int
    )
    basin_ids = np.asarray(
        info.get("basin_ids", np.arange(n, dtype=int)), dtype=int
    )

    return {
        "s": s,
        "F": F,
        "D_used": D_used,
        "kT": kT_val,
        "K": K_raw,
        "K_ps": K_ps,
        "exit_mean": exit_mean,
        "exit_ps": exit_mean * time_unit_ps,
        "k_out": k_out,
        "k_out_ps": k_out / time_unit_ps,
        "p_branch": p_branch,
        "labels_full": labels_full,
        "basin_ids": basin_ids,
    }


def _call_ctmc_1d(s, F, basin_network, D, beta, init_weight, verbose):
    """Call compute_ctmc_generator_fpe_1d and normalise output to (K, info)."""
    res = compute_ctmc_generator_fpe_1d(
        s=s,
        F=F,
        basin_network=basin_network,
        D=D,
        beta=beta,
        init_weight=init_weight,
        verbose=verbose,
    )
    if isinstance(res, dict):
        K = np.asarray(res["K"], dtype=float)
        info = {k: v for k, v in res.items() if k != "K"}
    else:  # legacy tuple
        K = np.asarray(res[0], dtype=float)
        info = res[1] if len(res) > 1 else {}
    return K, info


# ---------------------------------------------------------------------------
# D-profile helpers (ported from CLI script; reusable by external callers)
# ---------------------------------------------------------------------------
def interface_to_centers(
    x_interface: np.ndarray,
    D_interface: np.ndarray,
    method: str = "harmonic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Hummer-style interface diffusion values to bin-center values.

    Hummer's Bayesian estimator places D_{i+1/2} at internal bin *interfaces*.
    This routine reconstructs bin-center D_i by extrapolating one bin to each
    edge and then averaging (or taking harmonic means) of adjacent interface
    values.

    Parameters
    ----------
    x_interface : array (M,)
        CV positions of the M internal interfaces.
    D_interface : array (M,)
        D values at those interfaces.
    method : {'harmonic', 'avg'}
        ``'harmonic'`` (default) gives a better approximation for barriers
        (avoids over-estimating D at low-D regions); ``'avg'`` is simpler.

    Returns
    -------
    x_center : array (M+1,)
    D_center : array (M+1,)
    edges    : array (M+2,)  full edge positions used internally
    """
    xI = np.asarray(x_interface, dtype=float).ravel()
    DI = np.asarray(D_interface, dtype=float).ravel()
    if xI.size != DI.size:
        raise ValueError("x_interface and D_interface must have the same length.")
    if xI.size < 2:
        raise ValueError("Need at least 2 interface points.")

    dx_l = xI[1] - xI[0]
    dx_r = xI[-1] - xI[-2]
    edges = np.empty(xI.size + 2, dtype=float)
    edges[1:-1] = xI
    edges[0] = xI[0] - dx_l
    edges[-1] = xI[-1] + dx_r

    xC = 0.5 * (edges[:-1] + edges[1:])  # shape (M+1,)
    DC = np.empty(xC.size, dtype=float)
    DC[0] = DI[0]
    DC[-1] = DI[-1]

    if xC.size > 2:
        if method == "harmonic":
            with np.errstate(divide="ignore", invalid="ignore"):
                DC[1:-1] = 2.0 / (1.0 / DI[:-1] + 1.0 / DI[1:])
            bad = ~np.isfinite(DC[1:-1]) | (DC[1:-1] <= 0)
            DC[1:-1][bad] = 0.5 * (DI[:-1][bad] + DI[1:][bad])
        elif method == "avg":
            DC[1:-1] = 0.5 * (DI[:-1] + DI[1:])
        else:
            raise ValueError("method must be 'harmonic' or 'avg'.")

    return xC, DC, edges


def interpolate_D_to_grid(
    s_grid: np.ndarray,
    x_D: np.ndarray,
    D_vals: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a D(s) profile onto *s_grid*, clamping outside the range.

    Parameters
    ----------
    s_grid : array (N,)
        Target uniform grid.
    x_D : array (M,)
        CV positions of the diffusion profile.
    D_vals : array (M,)
        Diffusion values at those positions.
    method : {'linear', 'pchip'}
        Interpolation method.  ``'pchip'`` requires SciPy.

    Returns
    -------
    D_grid : array (N,)  non-negative finite diffusion values on s_grid.
    """
    s_grid = np.asarray(s_grid, dtype=float).ravel()
    x_D = np.asarray(x_D, dtype=float).ravel()
    D_vals = np.asarray(D_vals, dtype=float).ravel()

    good = D_vals[np.isfinite(D_vals) & (D_vals > 0)]
    if good.size == 0:
        raise ValueError("D profile has no positive finite values.")
    left = float(np.median(good[: min(3, good.size)]))
    right = float(np.median(good[max(0, good.size - 3):]))

    if method == "linear":
        Dg = np.interp(s_grid, x_D, D_vals, left=left, right=right)
    elif method == "pchip":
        from scipy.interpolate import PchipInterpolator  # type: ignore
        f = PchipInterpolator(x_D, D_vals, extrapolate=True)
        Dg = np.asarray(f(s_grid), dtype=float)
        Dg = np.where(s_grid < x_D[0], left, Dg)
        Dg = np.where(s_grid > x_D[-1], right, Dg)
    else:
        raise ValueError("method must be 'linear' or 'pchip'.")

    floor = float(good.min() * 1e-6)
    Dg = np.where(np.isfinite(Dg) & (Dg > 0), Dg, floor)
    return Dg


# ===========================================================================
# Public high-level workflow functions
# ===========================================================================

def run_1d_ctmc(
    s: Union[np.ndarray, Sequence],
    F: Union[np.ndarray, Sequence],
    D: Union[float, np.ndarray],
    T: float = 300.0,
    time_unit: str = "ps",
    max_basins: Optional[int] = None,
    core_fraction: Optional[float] = 0.05,
    init_weight: str = "boltzmann",
    resample_n: int = 500,
    verbose: bool = True,
) -> dict:
    """Compute 1D CTMC kinetics from arrays *s*, *F*, *D*.

    This is the lowest-level workflow: it accepts the free-energy profile and
    diffusion coefficient directly as arrays and handles everything else
    (basin detection, core labelling, BVP solve, result packing).

    Parameters
    ----------
    s : array (N,)
        *Uniform* grid of CV values.
    F : array (N,)
        Free energy in kJ/mol at each grid point.
    D : float or array (N,)
        Diffusion coefficient in CV²/time_unit.  Scalar = constant D;
        array = position-dependent D(s).
    T : float
        Temperature in Kelvin.  Default 300 K.
    time_unit : str
        Time unit of *D* (and of the returned ``K`` / ``exit_mean``).
        One of ``'fs'``, ``'ps'`` (default), ``'ns'``, ``'us'``.
    max_basins : int, optional
        Cap the number of detected basins.
    core_fraction : float, optional
        Fraction of each basin (ranked by proximity to the minimum) that
        counts as the "core" for CTMC entry/exit averaging.  0.05 = 5 %.
        Use ``None`` to use the full basin.
    init_weight : {'boltzmann', 'uniform'}
        Weight for basin averages of exit times.
    verbose : bool

    Returns
    -------
    dict
        See module-level docstring for key descriptions.
    """
    s = np.asarray(s, dtype=float).ravel()
    F = np.asarray(F, dtype=float).ravel()
    if s.size != F.size:
        raise ValueError("s and F must have the same length.")

    kT_val = _kT(T)
    beta = 1.0 / kT_val
    tps = _time_unit_to_ps(time_unit)

    bn = build_basin_network_from_fes_1d(s, F, max_basins=max_basins, verbose=verbose)
    _build_core_labels(s, F, bn, core_fraction)

    K, info = _call_ctmc_1d(s, F, bn, D, beta, init_weight, verbose)
    info["labels_full"] = bn.labels
    info["basin_ids"] = np.array([b.id for b in bn.basins], dtype=int)

    return _pack_result(s, F, D, kT_val, tps, K, info)


def run_1d_ctmc_from_plumed(
    fes_path: Union[str, Path],
    D: Union[float, np.ndarray],
    T: float = 300.0,
    time_unit: str = "ps",
    crop: Optional[Tuple[float, float]] = None,
    resample_n: Optional[int] = None,
    s_col: int = 0,
    F_col: int = 1,
    max_basins: Optional[int] = None,
    core_fraction: Optional[float] = 0.05,
    init_weight: str = "boltzmann",
    verbose: bool = True,
) -> dict:
    """Load a 1D PLUMED FES and compute CTMC kinetics with a constant *D*.

    Parameters
    ----------
    fes_path : str or Path
        Path to a PLUMED-style 1D FES file (comments start with ``#``).
    D : float
        Diffusion coefficient in CV²/time_unit.
    T : float
        Temperature in K.
    time_unit : str
        Time unit for *D* and for the returned rates.
    crop : (s_lo, s_hi), optional
        Restrict the FES to this CV range before analysis.
    resample_n : int, optional
        Resample to this many uniform points (useful after cropping).
    s_col, F_col : int
        Column indices for CV and free energy in the FES file.
    resample_n : int
        Resample the MFEP arc-length profile to this many uniform points
        before running the FPE solver (which requires a uniform grid).
        Default 500.
    max_basins, core_fraction, init_weight, verbose :
        Forwarded to :func:`run_1d_ctmc`.

    Returns
    -------
    dict  (same keys as :func:`run_1d_ctmc`)
    """
    from .fes import load_plumed_fes_1d

    s, F = load_plumed_fes_1d(
        fes_path, x_col=s_col, fes_col=F_col, verbose=verbose
    )

    if crop is not None:
        lo, hi = float(crop[0]), float(crop[1])
        mask = (s >= lo) & (s <= hi)
        s, F = s[mask], F[mask]

    if resample_n is not None:
        su = np.linspace(s[0], s[-1], int(resample_n))
        F = np.interp(su, s, F)
        s = su

    return run_1d_ctmc(
        s=s, F=F, D=D, T=T, time_unit=time_unit,
        max_basins=max_basins, core_fraction=core_fraction,
        init_weight=init_weight, verbose=verbose,
    )


def run_1d_ctmc_with_hummer_D(
    fes_path: Union[str, Path],
    d_csv: Union[str, Path],
    T: float = 300.0,
    time_unit: str = "ps",
    d_xcol: str = "x_interface",
    d_col: str = "D_med",
    d_grid: str = "interface",
    d_interface_mode: str = "harmonic",
    d_time_unit: str = "ps",
    d_interp_method: str = "linear",
    crop: Optional[Tuple[float, float]] = None,
    resample_n: int = 500,
    s_col: int = 0,
    F_col: int = 1,
    max_basins: Optional[int] = None,
    core_fraction: Optional[float] = 0.05,
    init_weight: str = "boltzmann",
    verbose: bool = True,
) -> dict:
    """Load a 1D PLUMED FES and a Hummer-style D(s) profile; compute CTMC.

    This is the standard workflow when D(s) has been estimated from short
    MD runs using the Hummer (2005) Bayesian inference.

    Parameters
    ----------
    fes_path : str or Path
        PLUMED 1D FES file.
    d_csv : str or Path
        CSV file with the diffusion profile.  Must contain at least two
        columns named *d_xcol* and *d_col*.
    T : float
        Temperature in K.
    time_unit : str
        Time unit for the returned rates.
    d_xcol : str
        Column name for CV positions in the CSV  (default ``'x_interface'``).
    d_col : str
        Column name for D values in the CSV  (default ``'D_med'``).
    d_grid : {'interface', 'center'}
        ``'interface'``: the CSV positions are internal bin *interfaces* and
        D values are face-centred (Hummer-style).  They will be converted to
        bin-centre values via :func:`interface_to_centers`.
        ``'center'``: D values are already at bin centres; no conversion.
    d_interface_mode : {'harmonic', 'avg'}
        Conversion method when ``d_grid='interface'``.
    d_time_unit : str
        Time unit of the raw D values in the CSV  (default ``'ps'``).
    d_interp_method : {'linear', 'pchip'}
        How to interpolate D onto the FES grid.
    crop : (s_lo, s_hi), optional
        CV range to retain from the FES.
    resample_n : int
        Resample the FES to this many uniform points before interpolating D.
        Required because the FPE solver needs a uniform grid.  Default 500.
    s_col, F_col : int
        Column indices in the FES file.
    resample_n : int
        Resample the MFEP arc-length profile to this many uniform points
        before running the FPE solver (which requires a uniform grid).
        Default 500.
    max_basins, core_fraction, init_weight, verbose :
        Forwarded to :func:`run_1d_ctmc`.

    Returns
    -------
    dict  (same keys as :func:`run_1d_ctmc`)
    """
    import pandas as pd  # local import; pandas is an optional dep
    from .fes import load_plumed_fes_1d

    # Load FES
    s, F = load_plumed_fes_1d(
        fes_path, x_col=s_col, fes_col=F_col, verbose=verbose
    )
    if crop is not None:
        lo, hi = float(crop[0]), float(crop[1])
        mask = (s >= lo) & (s <= hi)
        s, F = s[mask], F[mask]

    # Resample to uniform grid (mandatory for FPE solver)
    s_grid = np.linspace(float(s[0]), float(s[-1]), int(resample_n))
    F_grid = np.interp(s_grid, s, F)

    # Load D profile
    df = pd.read_csv(d_csv)
    x_D_raw = df[d_xcol].values.astype(float)
    D_raw = df[d_col].values.astype(float)

    # Interface → center conversion
    if d_grid == "interface":
        x_D_src, D_src, _ = interface_to_centers(
            x_D_raw, D_raw, method=d_interface_mode
        )
    else:
        x_D_src, D_src = x_D_raw, D_raw

    # Unit conversion: raw D [cv²/d_time_unit] → [cv²/time_unit]
    ps_per_d = _time_unit_to_ps(d_time_unit)
    ps_per_out = _time_unit_to_ps(time_unit)
    D_src = D_src * (ps_per_out / ps_per_d)

    # Interpolate D onto FES grid
    D_grid = interpolate_D_to_grid(
        s_grid, x_D_src, D_src, method=d_interp_method
    )

    return run_1d_ctmc(
        s=s_grid, F=F_grid, D=D_grid, T=T, time_unit=time_unit,
        max_basins=max_basins, core_fraction=core_fraction,
        init_weight=init_weight, verbose=verbose,
    )


def run_mfep_ctmc(
    fes2d_path: Union[str, Path],
    start: Tuple[float, float],
    end: Tuple[float, float],
    D_s: float,
    T: float = 300.0,
    time_unit: str = "ps",
    neb_images: int = 120,
    neb_steps: int = 3000,
    neb_spring: float = 1.0,
    use_neb: bool = True,
    max_basins: Optional[int] = None,
    core_fraction: Optional[float] = 0.05,
    init_weight: str = "boltzmann",
    resample_n: int = 500,
    verbose: bool = True,
) -> dict:
    """Find the MFEP on a 2D FES and compute 1D CTMC kinetics along it.

    Workflow:

    1. Load the 2D PLUMED FES.
    2. Find the minimum free-energy path (MFEP) from *start* to *end* using
       a grid-Dijkstra seed followed by NEB refinement.
    3. Extract the 1D free-energy profile F(s) along the arc-length s.
    4. Run :func:`run_1d_ctmc` on F(s) with the supplied constant *D_s*.

    Parameters
    ----------
    fes2d_path : str or Path
        Path to a 2D PLUMED FES file.
    start, end : (x, y)
        Start and end points for the MFEP in CV space.
    D_s : float
        Diffusion coefficient along the arc-length s  [arc-length²/time_unit].
    T : float
        Temperature in K.
    time_unit : str
        Time unit for *D_s* and the returned rates.
    neb_images : int
        Number of NEB images (controls resolution of the 1D profile).
    neb_steps : int
        Maximum NEB optimisation steps.
    neb_spring : float
        NEB spring constant (kJ/mol per CV² or compatible units).
    use_neb : bool
        If False, skip NEB refinement and use the raw grid path.
    resample_n : int
        Resample the MFEP arc-length profile to this many uniform points
        before running the FPE solver (which requires a uniform grid).
        Default 500.
    max_basins, core_fraction, init_weight, verbose :
        Forwarded to :func:`run_1d_ctmc`.

    Returns
    -------
    dict
        Same keys as :func:`run_1d_ctmc` plus ``'mfep_path'``
        (:class:`~stochkin.mfep.MFEPPath`).
    """
    # Load 2D FES
    x_grid, y_grid, fes_grid = load_plumed_fes_2d(fes2d_path, verbose=verbose)

    # MFEP (grid Dijkstra seed + optional NEB refinement)
    path = compute_mfep_profile_1d(
        x_grid=x_grid,
        y_grid=y_grid,
        fes_grid=fes_grid,
        start=start,
        end=end,
        use_neb=use_neb,
        neb_images=neb_images,
        neb_k_spring=neb_spring,
        neb_max_iter=neb_steps,
    )

    # Report NEB convergence
    if verbose and hasattr(path, "metadata") and "converged" in path.metadata:
        md = path.metadata
        conv_tag = "converged" if md["converged"] else "NOT converged"
        print(f"[NEB] {conv_tag} after {md['n_iter']} iters "
              f"(max_force={md['final_max_force']:.3e}, tol={md['tol']:.1e})")

    # 1D profile along arc-length – resample to uniform grid
    s_raw = path.s
    F_raw = path.F - np.nanmin(path.F)
    s = np.linspace(s_raw[0], s_raw[-1], int(resample_n))
    F_1d = np.interp(s, s_raw, F_raw)

    result = run_1d_ctmc(
        s=s, F=F_1d, D=D_s, T=T, time_unit=time_unit,
        max_basins=max_basins, core_fraction=core_fraction,
        init_weight=init_weight, verbose=verbose,
    )
    result["mfep_path"] = path
    return result


# ---------------------------------------------------------------------------
# Multi-MFEP workflow: all-basin pairwise rates from a 2D FES
# ---------------------------------------------------------------------------
def run_multi_mfep_ctmc(
    fes2d_path: Union[str, Path],
    D_s: float,
    T: float = 300.0,
    time_unit: str = "ps",
    neb_images: int =8120,
    neb_steps: int = 3000,
    neb_spring: float = 1.0,
    use_neb: bool = True,
    max_basins: Optional[int] = None,
    core_fraction: Optional[float] = 0.05,
    init_weight: str = "boltzmann",
    resample_n: int = 500,
    verbose: bool = True,
) -> dict:
    """Detect all basins on a 2D FES and compute pairwise MFEP-based CTMC rates.

    Workflow:

    1. Load the 2D PLUMED FES and build an interpolated potential.
    2. Detect all basins on the 2D surface.
    3. For every pair of basins (i, j), compute the MFEP connecting their
       minima and run the 1D CTMC pipeline on the resulting F(s) profile.
    4. Assemble a full N×N rate matrix from the pairwise 2-basin results.

    Parameters
    ----------
    fes2d_path : str or Path
        Path to a 2D PLUMED FES file.
    D_s : float
        Constant diffusion coefficient along the arc-length
        [arc-length² / time_unit].
    T : float
        Temperature in K.
    time_unit : str
        Time unit for rates (``'ps'``, ``'ns'``, …).
    neb_images, neb_steps, neb_spring, use_neb :
        NEB parameters forwarded to :func:`compute_mfep_profile_1d`.
    max_basins : int or None
        Keep only the *max_basins* deepest minima.
    core_fraction, init_weight :
        Forwarded to :func:`run_1d_ctmc`.
    resample_n : int
        Uniform resampling density for F(s) before the FPE solve.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict with keys:
        basin_network : BasinNetwork
            The detected 2D basins.
        basin_ids : np.ndarray
            Sorted array of basin indices.
        K : np.ndarray, shape (N, N)
            Full rate matrix [1/time_unit].
        K_ps : np.ndarray
            Rate matrix in ps⁻¹.
        exit_times : np.ndarray
            Mean exit time from each basin [time_unit].
        exit_ps : np.ndarray
            Mean exit time in ps.
        kT : float
            Thermal energy [kJ/mol].
        legs : dict
            ``legs[(i, j)]`` is the full result dict from ``run_mfep_ctmc``
            for the directed MFEP leg from basin *i* to basin *j*.
    """
    # 1. Load the 2D FES
    x_grid, y_grid, fes_grid = load_plumed_fes_2d(fes2d_path, verbose=verbose)
    kT = _kT(T)
    ps_per_unit = _time_unit_to_ps(time_unit)

    # 2. Build an interpolated potential and detect 2D basins
    fes_pot = make_fes_potential_from_grid(x_grid, y_grid, fes_grid)
    basin_net = build_basin_network_from_potential(
        fes_pot,
        xlim=(float(x_grid[0]),  float(x_grid[-1])),
        ylim=(float(y_grid[0]),  float(y_grid[-1])),
        nx=len(x_grid),
        ny=len(y_grid),
        max_basins=max_basins,
        verbose=verbose,
    )

    n = basin_net.n_basins
    basin_ids = np.array([b.id for b in basin_net.basins])
    if verbose:
        for b in basin_net.basins:
            print(f"  Basin {b.id}: minimum at ({b.minimum[0]:.2f}, "
                  f"{b.minimum[1]:.2f}), F_min={b.f_min:.2f}")

    # 3. Pairwise MFEPs
    K = np.zeros((n, n), dtype=float)
    legs: dict = {}

    for i, j in combinations(range(n), 2):
        bi, bj = basin_net.basins[i], basin_net.basins[j]
        start = tuple(bi.minimum.tolist())
        end = tuple(bj.minimum.tolist())

        if verbose:
            print(f"\n--- MFEP leg {bi.id} → {bj.id}  "
                  f"({start} → {end}) ---")

        try:
            leg = run_mfep_ctmc(
                fes2d_path=fes2d_path,
                start=start,
                end=end,
                D_s=D_s,
                T=T,
                time_unit=time_unit,
                neb_images=neb_images,
                neb_steps=neb_steps,
                neb_spring=neb_spring,
                use_neb=use_neb,
                core_fraction=core_fraction,
                init_weight=init_weight,
                resample_n=resample_n,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(f"  ⚠  Leg {bi.id}→{bj.id} failed: {exc}")
            continue

        legs[(i, j)] = leg

        # The leg result contains a 2-basin K matrix on the *arc-length*
        # sub-problem. Basin 0 in the leg corresponds to the start (basin i)
        # and basin 1 corresponds to the end (basin j).
        K_leg = leg["K"]  # shape (n_leg, n_leg) in [1/time_unit]
        n_leg = K_leg.shape[0]

        if n_leg >= 2:
            # Forward rate: K_leg[0, 1] is k(i→j), K_leg[1, 0] is k(j→i)
            K[i, j] += K_leg[0, n_leg - 1]
            K[j, i] += K_leg[n_leg - 1, 0]
        elif verbose:
            print(f"  ⚠  Leg {bi.id}→{bj.id}: only {n_leg} basin(s) "
                  "detected along the MFEP — no rate extracted.")

    # 4. Fix diagonal so each row sums to 0
    for i in range(n):
        K[i, i] = -np.sum(K[i, :])

    K_ps = K * ps_per_unit
    with np.errstate(divide="ignore"):
        exit_times = np.where(K.diagonal() != 0, -1.0 / K.diagonal(), np.inf)
    exit_ps = exit_times * ps_per_unit

    if verbose:
        print(f"\n{'='*60}")
        print(f"Full {n}×{n} rate matrix K [{time_unit}⁻¹]:")
        print(K)
        print(f"\nExit times [{time_unit}]: {exit_times}")

    return {
        "basin_network": basin_net,
        "basin_ids": basin_ids,
        "K": K,
        "K_ps": K_ps,
        "exit_times": exit_times,
        "exit_ps": exit_ps,
        "kT": kT,
        "legs": legs,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # High-level wrappers
    "run_1d_ctmc",
    "run_1d_ctmc_from_plumed",
    "run_1d_ctmc_with_hummer_D",
    "run_mfep_ctmc",
    "run_multi_mfep_ctmc",
    # D-profile helpers exposed for advanced users
    "interface_to_centers",
    "interpolate_D_to_grid",
]
