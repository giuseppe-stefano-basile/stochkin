"""stochkin.uncertainty
=====================

Monte Carlo uncertainty propagation for CTMC kinetics on 1-D
free-energy surfaces.

**Strategy.**  Perturb the inputs *F(s)* and *D(s)* within their error
bars, re-run the CTMC pipeline for each perturbed sample, and collect
statistics (mean, standard deviation, percentile-based confidence
intervals) on every predicted quantity (rates, exit times, branching
probabilities).

Two perturbation models are provided:

* **Gaussian (additive)** – suitable for free energies *F(s)*.
* **Log-normal (multiplicative)** – suitable for diffusion coefficients
  *D(s)*, which must remain positive.

When the Hummer Bayesian estimator supplies posterior credible intervals
(``F_lo``/``F_hi``, ``D_lo``/``D_hi``), the module converts them to
standard deviations automatically.

Main entry points
-----------------
bootstrap_ctmc_1d
    Propagate *F* and *D* uncertainties through
    :func:`~stochkin.workflows.run_1d_ctmc`.
bootstrap_ctmc_with_hummer_D
    Convenience wrapper that reads Hummer posterior intervals from CSV
    and propagates them through
    :func:`~stochkin.workflows.run_1d_ctmc_with_hummer_D`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .workflows import (
    _kT,
    _time_unit_to_ps,
    run_1d_ctmc,
    run_1d_ctmc_with_hummer_D,
    interface_to_centers,
    interpolate_D_to_grid,
)


# ======================================================================
# Perturbation samplers
# ======================================================================

_Z95 = 1.959964  # z-score for 95 % two-sided CI


def _ci_to_sigma(lo: np.ndarray, hi: np.ndarray,
                 ci_level: float = 0.95) -> np.ndarray:
    """Convert a symmetric credible interval [lo, hi] to a Gaussian σ."""
    from scipy.stats import norm  # type: ignore
    z = norm.ppf(0.5 + ci_level / 2.0)
    return np.maximum((np.asarray(hi) - np.asarray(lo)) / (2.0 * z), 0.0)


def _ci_to_log_sigma(lo: np.ndarray, hi: np.ndarray,
                     ci_level: float = 0.95) -> np.ndarray:
    """Convert [D_lo, D_hi] to log-space σ for log-normal sampling.

    Assumes that (log D_lo, log D_hi) spans a ``ci_level`` credible
    interval on the log scale.
    """
    from scipy.stats import norm  # type: ignore
    z = norm.ppf(0.5 + ci_level / 2.0)
    lo_a = np.asarray(lo, dtype=float)
    hi_a = np.asarray(hi, dtype=float)
    # Protect against non-positive values
    lo_safe = np.where(lo_a > 0, lo_a, 1e-30)
    hi_safe = np.where(hi_a > 0, hi_a, 1e-30)
    return np.maximum((np.log(hi_safe) - np.log(lo_safe)) / (2.0 * z), 0.0)


def _smooth_noise(noise_1d: np.ndarray, s: np.ndarray,
                  corr_length: float) -> np.ndarray:
    """Apply Gaussian smoothing to *noise_1d* with a given correlation length.

    This turns i.i.d. perturbations into spatially correlated noise with
    approximate correlation length *corr_length* in CV units, while
    preserving the point-wise variance (approximately).
    """
    ds = float(np.median(np.diff(s)))
    if corr_length <= ds:
        return noise_1d
    sigma_pts = corr_length / ds
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore
    except ImportError:
        # Fallback: simple running-average box filter
        k = max(1, int(round(sigma_pts)))
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        smoothed = np.convolve(noise_1d, kernel, mode="same")
        # Rescale to preserve variance
        scale = np.std(noise_1d) / max(np.std(smoothed), 1e-30)
        return smoothed * scale

    smoothed = gaussian_filter1d(noise_1d, sigma=sigma_pts, mode="reflect")
    # Rescale to preserve variance
    scale = np.std(noise_1d) / max(np.std(smoothed), 1e-30)
    return smoothed * scale


def _sample_F(
    rng: np.random.RandomState,
    F: np.ndarray,
    F_sigma: Optional[np.ndarray],
    s: np.ndarray,
    corr_length: Optional[float],
) -> np.ndarray:
    """Draw one perturbed F(s) sample (additive Gaussian).

    When *corr_length* is ``None`` a minimal smoothing of 3 grid
    spacings is applied so that i.i.d. noise does not introduce
    spurious local minima (which would change the basin topology).
    """
    if F_sigma is None:
        return F.copy()
    ds = float(np.median(np.diff(s)))
    # Default smoothing: 3 grid spacings (avoids spurious minima)
    cl = corr_length if corr_length is not None else 3.0 * ds
    noise = _smooth_noise(rng.standard_normal(F.size), s, cl) * F_sigma
    return F + noise


def _sample_D(
    rng: np.random.RandomState,
    D: np.ndarray,
    D_log_sigma: Optional[np.ndarray],
    D_sigma: Optional[np.ndarray],
    s: np.ndarray,
    corr_length: Optional[float],
) -> np.ndarray:
    """Draw one perturbed D(s) sample (log-normal or Gaussian, always > 0).

    As with *_sample_F*, a minimal smoothing of 3 grid spacings is
    applied when *corr_length* is ``None``.
    """
    ds = float(np.median(np.diff(s)))
    cl = corr_length if corr_length is not None else 3.0 * ds

    if D_log_sigma is not None:
        noise = _smooth_noise(rng.standard_normal(D.size), s, cl)
        noise /= max(np.std(noise), 1e-30)  # unit variance
        D_samp = D * np.exp(noise * D_log_sigma)
    elif D_sigma is not None:
        noise = _smooth_noise(rng.standard_normal(D.size), s, cl)
        noise /= max(np.std(noise), 1e-30)
        D_samp = D + noise * D_sigma
    else:
        return D.copy()
    # Ensure positivity
    floor = float(np.min(D[D > 0]) * 1e-6) if np.any(D > 0) else 1e-30
    return np.maximum(D_samp, floor)


# ======================================================================
# Result container
# ======================================================================

@dataclass
class UncertaintyResult:
    """Container for bootstrap uncertainty estimates.

    All ``*_mean``, ``*_std``, ``*_ci_lo``, ``*_ci_hi`` arrays have the
    same shape as the corresponding quantity in the reference (unperturbed)
    result.  The ``*_samples`` arrays have an extra leading axis of size
    ``n_bootstrap``.

    Attributes
    ----------
    reference : dict
        Full result dictionary from the *unperturbed* run
        (see :func:`~stochkin.workflows.run_1d_ctmc`).
    n_bootstrap : int
        Number of successful bootstrap replicates.
    n_failed : int
        Number of failed replicates (basin detection changed, solver
        diverged, etc.).
    confidence_level : float
        Confidence level for the reported CI (default 0.95).
    K_mean, K_std, K_ci_lo, K_ci_hi : ndarray
        Statistics of the rate matrix *K* [1/time_unit].
    K_ps_mean, K_ps_std, K_ps_ci_lo, K_ps_ci_hi : ndarray
        Statistics of *K* in ps⁻¹.
    exit_mean_mean, exit_mean_std, exit_mean_ci_lo, exit_mean_ci_hi : ndarray
        Statistics of the mean exit time per basin [time_unit].
    k_out_mean, k_out_std, k_out_ci_lo, k_out_ci_hi : ndarray
        Statistics of the total exit rate per basin [1/time_unit].
    p_branch_mean, p_branch_std, p_branch_ci_lo, p_branch_ci_hi : ndarray
        Statistics of the branching-probability matrix.
    K_samples : ndarray, shape (n_bootstrap, n_basins, n_basins)
        Raw bootstrap samples of *K*.
    exit_mean_samples : ndarray, shape (n_bootstrap, n_basins)
        Raw samples of exit times.
    k_out_samples : ndarray, shape (n_bootstrap, n_basins)
        Raw samples of exit rates.
    p_branch_samples : ndarray, shape (n_bootstrap, n_basins, n_basins)
        Raw samples of branching probabilities.
    """
    reference: dict
    n_bootstrap: int
    n_failed: int
    confidence_level: float

    # Rate matrix
    K_mean: np.ndarray
    K_std: np.ndarray
    K_ci_lo: np.ndarray
    K_ci_hi: np.ndarray
    K_samples: np.ndarray

    K_ps_mean: np.ndarray
    K_ps_std: np.ndarray
    K_ps_ci_lo: np.ndarray
    K_ps_ci_hi: np.ndarray

    # Exit times
    exit_mean_mean: np.ndarray
    exit_mean_std: np.ndarray
    exit_mean_ci_lo: np.ndarray
    exit_mean_ci_hi: np.ndarray
    exit_mean_samples: np.ndarray

    # Exit rates
    k_out_mean: np.ndarray
    k_out_std: np.ndarray
    k_out_ci_lo: np.ndarray
    k_out_ci_hi: np.ndarray
    k_out_samples: np.ndarray

    # Branching probabilities
    p_branch_mean: np.ndarray
    p_branch_std: np.ndarray
    p_branch_ci_lo: np.ndarray
    p_branch_ci_hi: np.ndarray
    p_branch_samples: np.ndarray

    def summary(self, time_unit: str = "") -> str:
        """Return a human-readable summary string."""
        tu = f" [{time_unit}]" if time_unit else ""
        tu_inv = f" [{time_unit}⁻¹]" if time_unit else ""
        lines = [
            f"Bootstrap CTMC uncertainty  ({self.n_bootstrap} replicates, "
            f"{self.n_failed} failed, {self.confidence_level:.0%} CI)",
            "",
        ]
        nb = self.K_mean.shape[0]
        # Per-basin exit times and rates
        lines.append(f"{'Basin':>6}  {'<τ_exit>' + tu:>22}  "
                     f"{'k_out' + tu_inv:>22}")
        lines.append(f"{'':>6}  {'mean ± std':>22}  {'mean ± std':>22}")
        for i in range(nb):
            tau_s = (f"{self.exit_mean_mean[i]:.4g} ± "
                     f"{self.exit_mean_std[i]:.2g}")
            k_s = (f"{self.k_out_mean[i]:.4g} ± "
                   f"{self.k_out_std[i]:.2g}")
            lines.append(f"{i:>6}  {tau_s:>22}  {k_s:>22}")

        lines.append("")
        lines.append(f"Rate matrix K{tu_inv} (mean ± std):")
        for i in range(nb):
            row = "  ".join(
                f"{self.K_mean[i, j]:+.3e}±{self.K_std[i, j]:.1e}"
                for j in range(nb)
            )
            lines.append(f"  [{row}]")

        lines.append("")
        lines.append(f"Rate matrix K{tu_inv} ({self.confidence_level:.0%} CI):")
        for i in range(nb):
            row = "  ".join(
                f"[{self.K_ci_lo[i, j]:+.3e}, {self.K_ci_hi[i, j]:+.3e}]"
                for j in range(nb)
            )
            lines.append(f"  [{row}]")

        return "\n".join(lines)


# ======================================================================
# Aggregation helper
# ======================================================================

def _aggregate(
    samples: np.ndarray,
    confidence: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean, std, CI_lo, CI_hi from a stack of samples (axis 0)."""
    alpha = (1.0 - confidence) / 2.0
    with np.errstate(all="ignore"):
        mean = np.nanmean(samples, axis=0)
        std = np.nanstd(samples, axis=0, ddof=1)
        ci_lo = np.nanpercentile(samples, 100 * alpha, axis=0)
        ci_hi = np.nanpercentile(samples, 100 * (1 - alpha), axis=0)
    return mean, std, ci_lo, ci_hi


# ======================================================================
# Main bootstrap
# ======================================================================

def bootstrap_ctmc_1d(
    s: Union[np.ndarray, Sequence],
    F: Union[np.ndarray, Sequence],
    D: Union[float, np.ndarray],
    *,
    # --- Perturbation specification for F ---
    F_err: Union[None, float, np.ndarray] = None,
    F_lo: Optional[np.ndarray] = None,
    F_hi: Optional[np.ndarray] = None,
    # --- Perturbation specification for D ---
    D_err: Union[None, float, np.ndarray] = None,
    D_rel_err: Union[None, float, np.ndarray] = None,
    D_lo: Optional[np.ndarray] = None,
    D_hi: Optional[np.ndarray] = None,
    # --- Bootstrap / correlation parameters ---
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    confidence: Optional[float] = None,
    corr_length: Optional[float] = None,
    seed: Optional[int] = None,
    # --- CTMC parameters (forwarded to run_1d_ctmc) ---
    T: float = 300.0,
    time_unit: str = "ps",
    max_basins: Optional[int] = None,
    core_fraction: Optional[float] = 0.05,
    init_weight: str = "boltzmann",
    verbose: bool = False,
) -> UncertaintyResult:
    """Propagate F(s) / D(s) uncertainties through the 1-D CTMC pipeline.

    For each bootstrap replicate:

    1. Draw a perturbed F(s) from *N(F, σ_F)* (Gaussian, additive).
    2. Draw a perturbed D(s) from *LogNormal(D, σ_log D)* or
       *N(D, σ_D)* (always clamped to D > 0).
    3. Run :func:`~stochkin.workflows.run_1d_ctmc` on the perturbed
       inputs.

    If the number of detected basins changes for a given replicate
    (different topology), that replicate is discarded.

    Parameters
    ----------
    s, F, D : array-like
        Central (best-estimate) grid, free energy, and diffusion
        coefficient.  Same semantics as :func:`run_1d_ctmc`.

    F_err : float or array, optional
        Standard deviation of F(s).  Scalar → uniform error.
    F_lo, F_hi : array, optional
        Lower / upper bounds of a credible interval on F.  Used to
        compute σ_F when *F_err* is not given.

    D_err : float or array, optional
        *Absolute* standard deviation of D(s) (Gaussian perturbation).
    D_rel_err : float or array, optional
        *Relative* error of D(s) (e.g. 0.3 = 30 %).  Converted to a
        log-normal σ.
    D_lo, D_hi : array, optional
        Lower / upper bounds of a credible interval on D(s).  Converted
        to a log-normal σ when neither *D_err* nor *D_rel_err* is given.

    n_bootstrap : int
        Number of bootstrap replicates (default 200).
    ci_level : float
        Credible-interval level used to interpret *F_lo/F_hi* and
        *D_lo/D_hi* (default 0.95 → 95 % CI).
    confidence : float, optional
        Confidence level for the *output* intervals (defaults to
        *ci_level*).
    corr_length : float, optional
        Spatial correlation length (in CV units) for the perturbation
        noise.  When set, point-wise i.i.d. noise is smoothed by a
        Gaussian kernel of this width, producing correlated
        perturbations.  ``None`` → independent noise at each grid point.
    seed : int, optional
        Random seed for reproducibility.

    T, time_unit, max_basins, core_fraction, init_weight :
        Forwarded to :func:`run_1d_ctmc`.
    verbose : bool
        If ``True``, print a progress counter.

    Returns
    -------
    UncertaintyResult
        Dataclass with ``*_mean``, ``*_std``, ``*_ci_lo``, ``*_ci_hi``
        for every CTMC output, plus the full ``*_samples`` arrays and
        the *reference* (unperturbed) result.

    Examples
    --------
    >>> import numpy as np, stochkin as sk
    >>> s = np.linspace(0, 1, 200)
    >>> F = 5.0 * (1 - (2*s - 1)**2)**2; F -= F.min()
    >>> res = sk.bootstrap_ctmc_1d(s, F, D=0.01, F_err=0.5,
    ...                            n_bootstrap=50, seed=42)
    >>> print(res.summary("ps"))
    """
    s = np.asarray(s, dtype=float).ravel()
    F = np.asarray(F, dtype=float).ravel()
    n_grid = s.size

    if np.isscalar(D):
        D_arr = np.full(n_grid, float(D))
    else:
        D_arr = np.asarray(D, dtype=float).ravel()

    if confidence is None:
        confidence = ci_level

    # ---- resolve F perturbation ----
    F_sigma: Optional[np.ndarray] = None
    if F_err is not None:
        F_sigma = np.broadcast_to(
            np.asarray(F_err, dtype=float), n_grid
        ).copy()
    elif F_lo is not None and F_hi is not None:
        F_sigma = _ci_to_sigma(
            np.asarray(F_lo, dtype=float),
            np.asarray(F_hi, dtype=float),
            ci_level=ci_level,
        )

    # ---- resolve D perturbation ----
    D_log_sigma: Optional[np.ndarray] = None
    D_sigma: Optional[np.ndarray] = None

    if D_lo is not None and D_hi is not None:
        D_log_sigma = _ci_to_log_sigma(
            np.asarray(D_lo, dtype=float),
            np.asarray(D_hi, dtype=float),
            ci_level=ci_level,
        )
    elif D_rel_err is not None:
        # Relative error → log-space σ  (for small rel_err, σ_log ≈ rel_err)
        D_log_sigma = np.broadcast_to(
            np.asarray(D_rel_err, dtype=float), n_grid
        ).copy()
    elif D_err is not None:
        D_sigma = np.broadcast_to(
            np.asarray(D_err, dtype=float), n_grid
        ).copy()

    # ---- reference (unperturbed) run ----
    ref = run_1d_ctmc(
        s=s, F=F, D=D_arr, T=T, time_unit=time_unit,
        max_basins=max_basins, core_fraction=core_fraction,
        init_weight=init_weight, verbose=False,
    )
    nb_ref = len(ref["basin_ids"])

    # Force every bootstrap replicate to detect the same number of
    # basins as the reference run.  Without this constraint,
    # perturbation noise can create or destroy shallow local minima,
    # leading to a different basin topology and an unusable replicate.
    bootstrap_max_basins = nb_ref

    # ---- bootstrap loop ----
    rng = np.random.RandomState(seed)

    K_list = []
    K_ps_list = []
    exit_list = []
    k_out_list = []
    p_branch_list = []
    n_failed = 0

    for b in range(n_bootstrap):
        if verbose and (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"  bootstrap {b + 1}/{n_bootstrap} …")

        F_b = _sample_F(rng, F, F_sigma, s, corr_length)
        D_b = _sample_D(rng, D_arr, D_log_sigma, D_sigma, s, corr_length)

        try:
            res_b = run_1d_ctmc(
                s=s, F=F_b, D=D_b, T=T, time_unit=time_unit,
                max_basins=bootstrap_max_basins,
                core_fraction=core_fraction,
                init_weight=init_weight, verbose=False,
            )
        except Exception:
            n_failed += 1
            continue

        # Discard if basin topology still changed (shouldn't happen with
        # max_basins pinned, but guard against edge cases).
        if len(res_b["basin_ids"]) != nb_ref:
            n_failed += 1
            continue

        K_list.append(res_b["K"])
        K_ps_list.append(res_b["K_ps"])
        exit_list.append(res_b["exit_mean"])
        k_out_list.append(res_b["k_out"])
        p_branch_list.append(res_b["p_branch"])

    if len(K_list) == 0:
        raise RuntimeError(
            "All bootstrap replicates failed.  Likely the FES perturbation "
            "is too large relative to the barrier height, causing basin "
            "topology changes in every sample."
        )

    K_all = np.array(K_list)
    K_ps_all = np.array(K_ps_list)
    exit_all = np.array(exit_list)
    kout_all = np.array(k_out_list)
    pb_all = np.array(p_branch_list)

    n_ok = K_all.shape[0]

    K_m, K_s, K_lo, K_hi = _aggregate(K_all, confidence)
    Kp_m, Kp_s, Kp_lo, Kp_hi = _aggregate(K_ps_all, confidence)
    ex_m, ex_s, ex_lo, ex_hi = _aggregate(exit_all, confidence)
    ko_m, ko_s, ko_lo, ko_hi = _aggregate(kout_all, confidence)
    pb_m, pb_s, pb_lo, pb_hi = _aggregate(pb_all, confidence)

    return UncertaintyResult(
        reference=ref,
        n_bootstrap=n_ok,
        n_failed=n_failed,
        confidence_level=confidence,
        # K
        K_mean=K_m, K_std=K_s, K_ci_lo=K_lo, K_ci_hi=K_hi,
        K_samples=K_all,
        # K_ps
        K_ps_mean=Kp_m, K_ps_std=Kp_s, K_ps_ci_lo=Kp_lo, K_ps_ci_hi=Kp_hi,
        # exit
        exit_mean_mean=ex_m, exit_mean_std=ex_s,
        exit_mean_ci_lo=ex_lo, exit_mean_ci_hi=ex_hi,
        exit_mean_samples=exit_all,
        # k_out
        k_out_mean=ko_m, k_out_std=ko_s,
        k_out_ci_lo=ko_lo, k_out_ci_hi=ko_hi,
        k_out_samples=kout_all,
        # p_branch
        p_branch_mean=pb_m, p_branch_std=pb_s,
        p_branch_ci_lo=pb_lo, p_branch_ci_hi=pb_hi,
        p_branch_samples=pb_all,
    )


# ======================================================================
# Hummer-D convenience wrapper
# ======================================================================

def bootstrap_ctmc_with_hummer_D(
    fes_path: Union[str, Path],
    d_csv: Union[str, Path],
    *,
    # --- Hummer FES uncertainty ---
    fes_err_path: Optional[Union[str, Path]] = None,
    F_err: Union[None, float, np.ndarray] = None,
    F_lo_col: str = "F_lo",
    F_hi_col: str = "F_hi",
    # --- Hummer D uncertainty ---
    D_lo_col: str = "D_lo",
    D_hi_col: str = "D_hi",
    perturb_D: bool = True,
    perturb_F: bool = True,
    # --- Bootstrap parameters ---
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    confidence: Optional[float] = None,
    corr_length: Optional[float] = None,
    seed: Optional[int] = None,
    # --- Parameters forwarded to run_1d_ctmc_with_hummer_D ---
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
    verbose: bool = False,
) -> UncertaintyResult:
    """Propagate Hummer-posterior uncertainties through the 1-D CTMC pipeline.

    This is a convenience wrapper around :func:`bootstrap_ctmc_1d` that:

    1.  Loads the PLUMED 1-D FES and Hummer D-profile CSV.
    2.  Reads ``D_lo`` / ``D_hi`` columns (and optionally ``F_lo`` /
        ``F_hi``) from the CSV to construct per-point error bars.
    3.  Converts both inputs to the common uniform grid.
    4.  Calls :func:`bootstrap_ctmc_1d` with the resolved σ arrays.

    Parameters
    ----------
    fes_path : str or Path
        PLUMED 1-D FES file.
    d_csv : str or Path
        Hummer diffusion-profile CSV (must contain *d_xcol*, *d_col*,
        and – if *perturb_D* – *D_lo_col* and *D_hi_col*).
    fes_err_path : str or Path, optional
        Separate CSV with the FES credible interval.  If given, must
        contain columns ``x_center``, ``F_lo``, ``F_hi`` (or names set
        by *F_lo_col* / *F_hi_col*).  When the FES uncertainty lives
        in the same CSV as D, set this to the same file.
    F_err : float or array, optional
        Uniform FES standard deviation.  Overrides *fes_err_path*.
    perturb_D, perturb_F : bool
        Toggle individual perturbation channels (default both ``True``).
    n_bootstrap, ci_level, confidence, corr_length, seed :
        Bootstrap and perturbation parameters — see
        :func:`bootstrap_ctmc_1d`.
    T, time_unit, d_xcol, d_col, d_grid, d_interface_mode, d_time_unit,
    d_interp_method, crop, resample_n, s_col, F_col, max_basins,
    core_fraction, init_weight :
        Forwarded to the underlying CTMC pipeline (same semantics as
        :func:`~stochkin.workflows.run_1d_ctmc_with_hummer_D`).
    verbose : bool
        Print progress.

    Returns
    -------
    UncertaintyResult
    """
    from .fes import load_plumed_fes_1d

    if confidence is None:
        confidence = ci_level

    # ---- Load FES ----
    s_raw, F_raw = load_plumed_fes_1d(
        fes_path, x_col=s_col, fes_col=F_col, verbose=False,
    )
    if crop is not None:
        lo, hi = float(crop[0]), float(crop[1])
        mask = (s_raw >= lo) & (s_raw <= hi)
        s_raw, F_raw = s_raw[mask], F_raw[mask]

    s_grid = np.linspace(float(s_raw[0]), float(s_raw[-1]), int(resample_n))
    F_grid = np.interp(s_grid, s_raw, F_raw)

    # ---- Load D profile ----
    df = _read_csv(d_csv)
    x_D_raw = df[d_xcol]
    D_raw = df[d_col]

    if d_grid == "interface":
        x_D_src, D_src, _ = interface_to_centers(
            x_D_raw, D_raw, method=d_interface_mode
        )
    else:
        x_D_src, D_src = x_D_raw, D_raw

    # Unit conversion
    ps_per_d = _time_unit_to_ps(d_time_unit)
    ps_per_out = _time_unit_to_ps(time_unit)
    D_src = D_src * (ps_per_out / ps_per_d)

    D_grid = interpolate_D_to_grid(
        s_grid, x_D_src, D_src, method=d_interp_method
    )

    # ---- Resolve D uncertainty ----
    D_lo_grid = None
    D_hi_grid = None
    if perturb_D and D_lo_col in df and D_hi_col in df:
        D_lo_raw = df[D_lo_col]
        D_hi_raw = df[D_hi_col]
        if d_grid == "interface":
            _, D_lo_src, _ = interface_to_centers(
                x_D_raw, D_lo_raw, method=d_interface_mode
            )
            _, D_hi_src, _ = interface_to_centers(
                x_D_raw, D_hi_raw, method=d_interface_mode
            )
        else:
            D_lo_src = D_lo_raw
            D_hi_src = D_hi_raw
        D_lo_src *= (ps_per_out / ps_per_d)
        D_hi_src *= (ps_per_out / ps_per_d)

        D_lo_grid = interpolate_D_to_grid(
            s_grid, x_D_src, D_lo_src, method=d_interp_method
        )
        D_hi_grid = interpolate_D_to_grid(
            s_grid, x_D_src, D_hi_src, method=d_interp_method
        )

    # ---- Resolve F uncertainty ----
    F_sigma_grid = None
    if perturb_F:
        if F_err is not None:
            F_sigma_grid = np.broadcast_to(
                np.asarray(F_err, dtype=float), s_grid.size
            ).copy()
        elif fes_err_path is not None:
            df_f = _read_csv(fes_err_path)
            x_f = next(iter(df_f.values()))  # first col = CV
            fl = df_f[F_lo_col]
            fh = df_f[F_hi_col]
            sigma_f_raw = _ci_to_sigma(fl, fh, ci_level=ci_level)
            F_sigma_grid = np.interp(s_grid, x_f, sigma_f_raw)

    # ---- Call the core bootstrap ----
    return bootstrap_ctmc_1d(
        s=s_grid,
        F=F_grid,
        D=D_grid,
        F_err=F_sigma_grid,
        D_lo=D_lo_grid,
        D_hi=D_hi_grid,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        confidence=confidence,
        corr_length=corr_length,
        seed=seed,
        T=T,
        time_unit=time_unit,
        max_basins=max_basins,
        core_fraction=core_fraction,
        init_weight=init_weight,
        verbose=verbose,
    )


# ======================================================================
# Public API
# ======================================================================

__all__ = [
    "UncertaintyResult",
    "bootstrap_ctmc_1d",
    "bootstrap_ctmc_with_hummer_D",
]
