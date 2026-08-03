"""User-supplied memory-kernel tools (experimental).

.. warning::
   This development API is experimental.  Names, defaults, numerical methods,
   and result schemas may change between minor releases.  Prefer importing it
   through :mod:`stochkin.experimental.memory`.

This module deliberately does **not** estimate memory kernels from
trajectories.  Users must provide a matrix-valued kernel ``Sigma_t`` with
shape ``(n_times, n_states, n_states)`` and units ``time^-2``.  The time grid
for the kernel must use the same time unit as the Markovian generator ``K0``.

The default convention follows the rest of :mod:`stochkin`: row vectors evolve
as ``p(t) = p(0) T(t)``, generators have rows summing to zero, and supplied
memory kernels should also have rows summing to zero.

The same kernel can be used either for direct generalized master equation
propagation or for the memory-corrected CTMC workflow, where memory moments
define an effective grid generator before basin coarse-graining.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np


_MISSING_KERNEL_MESSAGE = (
    "stochkin does not estimate memory kernels in this workflow. "
    "Please provide Sigma_t with shape (n_times, n_states, n_states) "
    "(for grid workflows, n_states must equal n_grid) and units time^-2, "
    "plus memory_times in the same time unit as K0."
)


@dataclass
class MemoryKernelInput:
    """Validated user-supplied memory kernel.

    Attributes
    ----------
    times : ndarray, shape (n_times,)
        Kernel time grid, in the same time unit as ``K0``.
    Sigma : ndarray, shape (n_times, n_states, n_states)
        Matrix-valued memory kernel.  Units are ``time^-2``.
    convention : {"row", "column"}
        Matrix orientation convention.
    diagnostics : dict
        Shape, conservation, and finite-value diagnostics.
    """

    times: np.ndarray
    Sigma: np.ndarray
    convention: str
    diagnostics: Dict[str, Any]


@dataclass
class MemoryValidationResult:
    """Result returned by :func:`validate_memory_kernel`."""

    is_valid: bool
    times: np.ndarray
    Sigma: np.ndarray
    convention: str
    diagnostics: Dict[str, Any]


@dataclass
class GMEPropagationResult:
    """Container for probability or transition-matrix GME propagation."""

    times: np.ndarray
    trajectory: np.ndarray
    K0: np.ndarray
    Sigma: np.ndarray
    memory_times: np.ndarray
    convention: str
    diagnostics: Dict[str, Any]


@dataclass
class EffectiveGeneratorResult:
    """Memory-corrected effective generator and convergence diagnostics."""

    K_eff: np.ndarray
    order: int
    moments: Dict[int, np.ndarray]
    convention: str
    diagnostics: Dict[str, Any]


def _as_convention(convention: str) -> str:
    conv = str(convention).lower()
    if conv not in {"row", "column"}:
        raise ValueError("convention must be 'row' or 'column'")
    return conv


def _missing_kernel_if_needed(Sigma_t, times) -> None:
    if Sigma_t is None or times is None:
        raise ValueError(_MISSING_KERNEL_MESSAGE)


def _as_dense_array(A, name: str) -> np.ndarray:
    if hasattr(A, "toarray"):
        A = A.toarray()
    arr = np.asarray(A, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_matrix_stack(Sigma_t, times) -> Tuple[np.ndarray, np.ndarray]:
    _missing_kernel_if_needed(Sigma_t, times)
    Sigma = _as_dense_array(Sigma_t, "Sigma_t")
    t = np.asarray(times, dtype=float).ravel()

    if Sigma.ndim != 3:
        raise ValueError(
            _MISSING_KERNEL_MESSAGE
            + " Received Sigma_t with ndim={}; expected ndim=3.".format(Sigma.ndim)
        )
    if Sigma.shape[1] != Sigma.shape[2]:
        raise ValueError("Sigma_t matrices must be square")
    if t.ndim != 1 or t.size != Sigma.shape[0]:
        raise ValueError("memory_times must be 1D and match Sigma_t.shape[0]")
    if t.size < 1:
        raise ValueError("memory_times must contain at least one time point")
    if not np.all(np.isfinite(t)):
        raise ValueError("memory_times contains non-finite values")
    if t[0] < 0.0:
        raise ValueError("memory_times must start at or after zero")
    if t.size > 1 and not np.all(np.diff(t) > 0.0):
        raise ValueError("memory_times must be strictly increasing")
    return Sigma.copy(), t.copy()


def _axis_sums(A: np.ndarray, convention: str) -> np.ndarray:
    if convention == "row":
        return np.sum(A, axis=-1)
    return np.sum(A, axis=-2)


def enforce_generator_conservation(A, convention: str = "row") -> np.ndarray:
    """Return a copy of ``A`` projected to row- or column-sum zero.

    The projection changes only diagonal entries.  For row convention each
    matrix row is forced to sum to zero; for column convention each column is
    forced to sum to zero.  ``A`` may be a single square matrix or a stack with
    shape ``(..., n, n)``.
    """

    conv = _as_convention(convention)
    arr = _as_dense_array(A, "A").copy()
    if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError("A must be square on its last two axes")

    n = arr.shape[-1]
    idx = np.arange(n)
    sums = _axis_sums(arr, conv)
    arr[..., idx, idx] -= sums
    return arr


def validate_memory_kernel(
    Sigma_t,
    times,
    K0=None,
    *,
    convention: str = "row",
    enforce_conservation: bool = False,
) -> MemoryValidationResult:
    """Validate a user-supplied memory kernel.

    Parameters
    ----------
    Sigma_t : array_like, shape (n_times, n_states, n_states)
        Memory kernel supplied by the user.  Units must be ``time^-2``.
    times : array_like, shape (n_times,)
        Kernel time grid in the same time unit used for ``K0``.
    K0 : array_like, optional
        Markovian generator with units ``time^-1``.  If provided, its shape and
        conservation convention are checked against ``Sigma_t``.
    convention : {"row", "column"}
        Row convention checks row sums; column convention checks column sums.
    enforce_conservation : bool
        If True, return a copy of ``Sigma_t`` with diagonal entries adjusted so
        the requested sums are zero.
    """

    conv = _as_convention(convention)
    Sigma, t = _validate_matrix_stack(Sigma_t, times)

    sums_before = _axis_sums(Sigma, conv)
    max_abs_before = float(np.max(np.abs(sums_before))) if sums_before.size else 0.0

    Sigma_used = enforce_generator_conservation(Sigma, conv) if enforce_conservation else Sigma
    sums_after = _axis_sums(Sigma_used, conv)
    max_abs_after = float(np.max(np.abs(sums_after))) if sums_after.size else 0.0
    correction = Sigma_used - Sigma
    correction_norm = float(np.linalg.norm(correction.ravel()))
    base_norm = float(np.linalg.norm(Sigma.ravel()))

    diagnostics: Dict[str, Any] = {
        "n_times": int(t.size),
        "n_states": int(Sigma.shape[1]),
        "time_start": float(t[0]),
        "time_stop": float(t[-1]),
        "dt": float(t[1] - t[0]) if t.size > 1 else np.nan,
        "uniform_time_grid": bool(
            t.size < 3 or np.allclose(np.diff(t), t[1] - t[0], rtol=1e-9, atol=1e-12)
        ),
        "max_abs_conservation_sum_before": max_abs_before,
        "max_abs_conservation_sum_after": max_abs_after,
        "relative_conservation_correction": correction_norm / max(base_norm, 1e-300),
        "enforced_conservation": bool(enforce_conservation),
    }

    if K0 is not None:
        K = _as_dense_array(K0, "K0")
        if K.shape != Sigma.shape[1:]:
            raise ValueError(
                f"K0 has shape {K.shape}, but Sigma_t matrices have shape {Sigma.shape[1:]}"
            )
        k_sums = _axis_sums(K, conv)
        diagnostics["K0_max_abs_conservation_sum"] = float(np.max(np.abs(k_sums)))

    return MemoryValidationResult(
        is_valid=True,
        times=t,
        Sigma=Sigma_used,
        convention=conv,
        diagnostics=diagnostics,
    )


def _require_uniform_times(times: np.ndarray, name: str) -> float:
    times = np.asarray(times, dtype=float).ravel()
    if times.size < 2:
        raise ValueError(f"{name} must contain at least two points for propagation")
    dt = np.diff(times)
    if not np.all(dt > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    dt0 = float(dt[0])
    if not np.allclose(dt, dt0, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced for the v1 explicit solver")
    return dt0


def _prepare_kernel_on_dt(Sigma: np.ndarray, kernel_times: np.ndarray, dt: float) -> np.ndarray:
    kdt = _require_uniform_times(kernel_times, "memory_times")
    if not np.isclose(kdt, dt, rtol=1e-8, atol=1e-12):
        raise ValueError(
            "memory_times spacing must match propagation dt for the v1 explicit solver"
        )
    return Sigma


def _propagation_diagnostics(values: np.ndarray, convention: str, is_transition: bool) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    if is_transition:
        sums = _axis_sums(values, convention)
        diagnostics["max_abs_normalization_drift"] = float(np.max(np.abs(sums - 1.0)))
        diagnostics["minimum_probability"] = float(np.min(values))
        neg = values[values < 0.0]
        diagnostics["negative_mass"] = float(-np.sum(neg)) if neg.size else 0.0
        diagnostics["fraction_negative_entries"] = float(np.mean(values < 0.0))
    else:
        sums = np.sum(values, axis=1)
        diagnostics["max_abs_normalization_drift"] = float(np.max(np.abs(sums - 1.0)))
        diagnostics["minimum_probability"] = float(np.min(values))
        neg = values[values < 0.0]
        diagnostics["negative_mass"] = float(-np.sum(neg)) if neg.size else 0.0
        diagnostics["fraction_negative_entries"] = float(np.mean(values < 0.0))
    return diagnostics


def _generator_diagnostics(A: np.ndarray, convention: str) -> Dict[str, Any]:
    """Diagnostics for whether ``A`` resembles a conservative CTMC generator."""

    conv = _as_convention(convention)
    arr = _as_dense_array(A, "A")
    n = arr.shape[-1]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("A must be a square matrix")

    sums = _axis_sums(arr, conv)
    offdiag = arr.copy()
    np.fill_diagonal(offdiag, 0.0)
    diag = np.diag(arr)
    positive_tol = 1e-12
    min_offdiag = float(np.min(offdiag)) if offdiag.size else 0.0
    negative_offdiag_count = int(np.sum(offdiag < -positive_tol))
    positive_diagonal_count = int(np.sum(diag > positive_tol))
    max_abs_sum = float(np.max(np.abs(sums))) if sums.size else 0.0

    return {
        "n_states": int(n),
        "max_abs_conservation_sum": max_abs_sum,
        "min_offdiagonal": min_offdiag,
        "negative_offdiagonal_count": negative_offdiag_count,
        "positive_diagonal_count": positive_diagonal_count,
        "generator_like": bool(
            max_abs_sum <= 1e-8
            and negative_offdiag_count == 0
            and positive_diagonal_count == 0
        ),
    }


def propagate_gme(
    p0,
    K0,
    Sigma_t,
    times,
    n_steps: Optional[int] = None,
    dt: Optional[float] = None,
    *,
    convention: str = "row",
    normalize: bool = False,
    check_positivity: bool = True,
) -> GMEPropagationResult:
    """Propagate a probability vector with a supplied memory kernel.

    ``Sigma_t`` must be supplied by the user; this function does not infer or
    estimate it.  For v1, ``memory_times`` and the propagation grid must be
    uniform and have the same spacing.
    """

    conv = _as_convention(convention)
    validation = validate_memory_kernel(Sigma_t, times, K0, convention=conv)
    Sigma = validation.Sigma
    memory_times = validation.times
    K = _as_dense_array(K0, "K0")
    step = _require_uniform_times(memory_times, "memory_times") if dt is None else float(dt)
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("dt must be a finite positive value")
    Sigma = _prepare_kernel_on_dt(Sigma, memory_times, step)

    if n_steps is None:
        n_steps = int(memory_times.size - 1)
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    p = np.asarray(p0, dtype=float).ravel()
    n = K.shape[0]
    if p.size != n:
        raise ValueError(f"p0 has length {p.size}, expected {n}")
    if conv == "column":
        p = p.reshape(n, 1)
    elif p.ndim == 1:
        p = p.reshape(1, n)

    values = np.zeros((n_steps + 1, n), dtype=float)
    values[0] = p.ravel()

    for step_idx in range(n_steps):
        if conv == "row":
            markov = values[step_idx] @ K
            memory = np.zeros(n, dtype=float)
            max_m = min(step_idx, Sigma.shape[0] - 1)
            for m in range(1, max_m + 1):
                memory += values[step_idx - m] @ Sigma[m]
            next_p = values[step_idx] + step * (markov + step * memory)
        else:
            current = values[step_idx].reshape(n, 1)
            markov = (K @ current).ravel()
            memory = np.zeros(n, dtype=float)
            max_m = min(step_idx, Sigma.shape[0] - 1)
            for m in range(1, max_m + 1):
                memory += (Sigma[m] @ values[step_idx - m].reshape(n, 1)).ravel()
            next_p = values[step_idx] + step * (markov + step * memory)

        if normalize:
            total = float(np.sum(next_p))
            if total != 0.0 and np.isfinite(total):
                next_p = next_p / total
        values[step_idx + 1] = next_p

    out_times = np.arange(n_steps + 1, dtype=float) * step
    diagnostics = dict(validation.diagnostics)
    diagnostics.update(_propagation_diagnostics(values, conv, is_transition=False))
    diagnostics["checked_positivity"] = bool(check_positivity)

    return GMEPropagationResult(
        times=out_times,
        trajectory=values,
        K0=K,
        Sigma=Sigma,
        memory_times=memory_times,
        convention=conv,
        diagnostics=diagnostics,
    )


def propagate_gme_transition_matrix(
    K0,
    Sigma_t,
    times,
    n_steps: Optional[int] = None,
    dt: Optional[float] = None,
    *,
    convention: str = "row",
) -> GMEPropagationResult:
    """Propagate the GME transition matrix from ``T(0)=I``."""

    conv = _as_convention(convention)
    validation = validate_memory_kernel(Sigma_t, times, K0, convention=conv)
    Sigma = validation.Sigma
    memory_times = validation.times
    K = _as_dense_array(K0, "K0")
    step = _require_uniform_times(memory_times, "memory_times") if dt is None else float(dt)
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("dt must be a finite positive value")
    Sigma = _prepare_kernel_on_dt(Sigma, memory_times, step)

    if n_steps is None:
        n_steps = int(memory_times.size - 1)
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    n = K.shape[0]
    values = np.zeros((n_steps + 1, n, n), dtype=float)
    values[0] = np.eye(n)

    for step_idx in range(n_steps):
        if conv == "row":
            markov = values[step_idx] @ K
            memory = np.zeros((n, n), dtype=float)
            max_m = min(step_idx, Sigma.shape[0] - 1)
            for m in range(1, max_m + 1):
                memory += values[step_idx - m] @ Sigma[m]
            values[step_idx + 1] = values[step_idx] + step * (markov + step * memory)
        else:
            markov = K @ values[step_idx]
            memory = np.zeros((n, n), dtype=float)
            max_m = min(step_idx, Sigma.shape[0] - 1)
            for m in range(1, max_m + 1):
                memory += Sigma[m] @ values[step_idx - m]
            values[step_idx + 1] = values[step_idx] + step * (markov + step * memory)

    out_times = np.arange(n_steps + 1, dtype=float) * step
    diagnostics = dict(validation.diagnostics)
    diagnostics.update(_propagation_diagnostics(values, conv, is_transition=True))

    return GMEPropagationResult(
        times=out_times,
        trajectory=values,
        K0=K,
        Sigma=Sigma,
        memory_times=memory_times,
        convention=conv,
        diagnostics=diagnostics,
    )


def memory_moments(Sigma_t, times, max_order: int = 2) -> Dict[int, np.ndarray]:
    """Return moments ``M_k = integral t^k Sigma(t) dt`` by trapezoid rule."""

    Sigma, t = _validate_matrix_stack(Sigma_t, times)
    max_order = int(max_order)
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return {
        k: trapezoid(Sigma * (t[:, None, None] ** k), t, axis=0)
        for k in range(max_order + 1)
    }


def effective_markov_generator_from_memory(
    K0,
    Sigma_t,
    times,
    order: int = 0,
    convention: str = "row",
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
    damping: float = 0.5,
    enforce_conservation: bool = False,
    warn: bool = True,
    return_result: bool = False,
):
    """Build a short-memory effective Markov generator from supplied memory.

    Orders 0 and 1 use explicit formulas.  For ``order >= 2`` the truncated
    moment expansion is solved by damped fixed-point iteration.  By default
    this function returns only the corrected generator for backward
    compatibility.  Set ``return_result=True`` to receive an
    :class:`EffectiveGeneratorResult` with diagnostics.
    """

    conv = _as_convention(convention)
    K = _as_dense_array(K0, "K0")
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K0 must be a square matrix")
    order = int(order)
    if order < 0:
        raise ValueError("order must be >= 0")
    moments = memory_moments(Sigma_t, times, max_order=max(1, order))
    M0 = moments[0]
    if M0.shape != K.shape:
        raise ValueError(
            f"K0 has shape {K.shape}, but Sigma_t moments have shape {M0.shape}"
        )
    diagnostics: Dict[str, Any] = {
        "order": order,
        "method": "explicit" if order <= 1 else "fixed_point",
        "converged": True,
        "iterations": 0,
        "residual_norm": 0.0,
        "damping": float(damping),
        "tol": float(tol),
        "max_iter": int(max_iter),
    }

    if order == 0:
        Keff = K + M0
    elif order == 1:
        I = np.eye(K.shape[0])
        if conv == "row":
            Keff = np.linalg.solve((I + moments[1]).T, (K + M0).T).T
        else:
            Keff = np.linalg.solve(I + moments[1], K + M0)
    else:
        if not (0.0 < float(damping) <= 1.0):
            raise ValueError("damping must be in (0, 1]")
        if float(tol) <= 0.0:
            raise ValueError("tol must be > 0")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be >= 1")

        # Start from the exact order-1 result; it is usually much closer than K0.
        I = np.eye(K.shape[0])
        if conv == "row":
            current = np.linalg.solve((I + moments[1]).T, (K + M0).T).T
        else:
            current = np.linalg.solve(I + moments[1], K + M0)

        residual = np.inf
        converged = False
        for it in range(1, int(max_iter) + 1):
            power = np.eye(K.shape[0])
            candidate = K.copy()
            for n in range(order + 1):
                if n > 0:
                    power = power @ current
                coeff = ((-1.0) ** n) / float(math.factorial(n))
                if conv == "row":
                    candidate = candidate + coeff * (power @ moments[n])
                else:
                    candidate = candidate + coeff * (moments[n] @ power)

            next_K = (1.0 - float(damping)) * current + float(damping) * candidate
            if enforce_conservation:
                next_K = enforce_generator_conservation(next_K, conv)
            residual = float(np.linalg.norm((next_K - current).ravel()))
            if not (np.isfinite(residual) and np.all(np.isfinite(next_K))):
                current = next_K
                converged = False
                break
            current = next_K
            current_norm = float(np.linalg.norm(current.ravel()))
            if np.isfinite(current_norm) and residual <= float(tol) * max(1.0, current_norm):
                converged = True
                break

        Keff = current
        diagnostics.update(
            {
                "converged": bool(converged),
                "iterations": int(it),
                "residual_norm": float(residual),
            }
        )

    if enforce_conservation:
        Keff = enforce_generator_conservation(Keff, conv)

    if np.all(np.isfinite(Keff)):
        diagnostics.update(_generator_diagnostics(Keff, conv))
    else:
        diagnostics.update(
            {
                "n_states": int(K.shape[0]),
                "max_abs_conservation_sum": np.inf,
                "min_offdiagonal": -np.inf,
                "negative_offdiagonal_count": -1,
                "positive_diagonal_count": -1,
                "generator_like": False,
            }
        )
    if warn and not diagnostics["generator_like"]:
        warnings.warn(
            "Memory-corrected effective generator is not CTMC-like; "
            "inspect diagnostics before interpreting rates.",
            RuntimeWarning,
            stacklevel=2,
        )
    if warn and not diagnostics["converged"]:
        warnings.warn(
            "Higher-order memory effective-generator iteration did not converge.",
            RuntimeWarning,
            stacklevel=2,
        )

    result = EffectiveGeneratorResult(
        K_eff=Keff,
        order=order,
        moments=moments,
        convention=conv,
        diagnostics=diagnostics,
    )
    return result if return_result else result.K_eff


def memory_corrected_generator(
    K0,
    Sigma_t,
    times,
    *,
    order: int = 0,
    convention: str = "row",
    **kwargs,
) -> np.ndarray:
    """Alias for :func:`effective_markov_generator_from_memory`."""

    return effective_markov_generator_from_memory(
        K0, Sigma_t, times, order=order, convention=convention, **kwargs
    )


def chapman_kolmogorov_error(T_lag, lag_times, norm: str = "fro") -> Dict[str, Any]:
    """Compute Chapman-Kolmogorov errors for a stack of transition matrices."""

    T = _as_dense_array(T_lag, "T_lag")
    times = np.asarray(lag_times, dtype=float).ravel()
    if T.ndim != 3 or T.shape[1] != T.shape[2]:
        raise ValueError("T_lag must have shape (n_lags, n_states, n_states)")
    if times.size != T.shape[0]:
        raise ValueError("lag_times must match T_lag.shape[0]")

    errors = []
    rel_errors = []
    pairs = []
    for i in range(T.shape[0]):
        for j in range(T.shape[0]):
            target_time = times[i] + times[j]
            k = int(np.argmin(np.abs(times - target_time)))
            if not np.isclose(times[k], target_time, rtol=1e-8, atol=1e-12):
                continue
            diff = T[k] - T[i] @ T[j]
            err = float(np.linalg.norm(diff, ord=norm))
            ref = float(np.linalg.norm(T[k], ord=norm))
            errors.append(err)
            rel_errors.append(err / max(ref, 1e-300))
            pairs.append((i, j, k))

    return {
        "ck_error_absolute": np.asarray(errors, dtype=float),
        "ck_error_relative": np.asarray(rel_errors, dtype=float),
        "ck_error_by_lag_pair": pairs,
        "max_absolute_error": float(np.max(errors)) if errors else 0.0,
        "max_relative_error": float(np.max(rel_errors)) if rel_errors else 0.0,
    }


def _markov_transition_stack(K0: np.ndarray, times: np.ndarray, convention: str) -> np.ndarray:
    from scipy.linalg import expm  # type: ignore

    K = K0 if convention == "row" else K0
    return np.asarray([expm(float(t) * K) for t in times], dtype=float)


def validate_memory_model(
    T_reference,
    lag_times,
    K0,
    Sigma_t,
    memory_times,
    *,
    convention: str = "row",
) -> Dict[str, Any]:
    """Compare supplied-memory GME propagation to reference transition matrices."""

    conv = _as_convention(convention)
    T_ref = _as_dense_array(T_reference, "T_reference")
    lag_times_arr = np.asarray(lag_times, dtype=float).ravel()
    if T_ref.ndim != 3 or T_ref.shape[1] != T_ref.shape[2]:
        raise ValueError("T_reference must have shape (n_lags, n_states, n_states)")
    if lag_times_arr.size != T_ref.shape[0]:
        raise ValueError("lag_times must match T_reference.shape[0]")
    if lag_times_arr.size < 2:
        raise ValueError("Need at least two lag times for validation")
    dt = _require_uniform_times(lag_times_arr, "lag_times")

    K = _as_dense_array(K0, "K0")
    n_steps = int(round(float(lag_times_arr[-1] - lag_times_arr[0]) / dt))
    gme = propagate_gme_transition_matrix(
        K,
        Sigma_t,
        memory_times,
        n_steps=n_steps,
        dt=dt,
        convention=conv,
    )
    if not np.isclose(lag_times_arr[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("lag_times must start at zero for v1 validation")
    T_gme = gme.trajectory[: T_ref.shape[0]]
    T_markov = _markov_transition_stack(K, lag_times_arr, conv)

    diff_markov = T_ref - T_markov
    diff_gme = T_ref - T_gme
    markov_err = np.linalg.norm(diff_markov.reshape(T_ref.shape[0], -1), axis=1)
    gme_err = np.linalg.norm(diff_gme.reshape(T_ref.shape[0], -1), axis=1)
    ref_norm = np.linalg.norm(T_ref.reshape(T_ref.shape[0], -1), axis=1)

    return {
        "T_markov": T_markov,
        "T_gme": T_gme,
        "gme_result": gme,
        "frobenius_error_markov": markov_err,
        "frobenius_error_gme": gme_err,
        "relative_error_markov": markov_err / np.maximum(ref_norm, 1e-300),
        "relative_error_gme": gme_err / np.maximum(ref_norm, 1e-300),
        "ck": chapman_kolmogorov_error(T_ref, lag_times_arr),
        "gme_diagnostics": gme.diagnostics,
    }
