import numpy as np
import pytest
from scipy.linalg import expm

import stochkin as sk
from stochkin.experimental import memory as experimental_memory
from stochkin.potentials import Basin1D, BasinNetwork1D


def _small_generator():
    K = np.array(
        [
            [-0.40, 0.40],
            [0.25, -0.25],
        ],
        dtype=float,
    )
    return K


def test_experimental_memory_namespace_is_explicit_and_complete():
    assert experimental_memory.EXPERIMENTAL_API is True
    assert experimental_memory.EXPERIMENTAL_API_VERSION == "0.1"
    assert experimental_memory.validate_memory_kernel is sk.validate_memory_kernel
    assert experimental_memory.run_gme_1d is sk.run_gme_1d
    assert experimental_memory.run_memory_corrected_ctmc_1d is sk.run_memory_corrected_ctmc_1d


def _memory_stack(times, amp=0.05, tau=0.08):
    base = np.array([[-1.0, 1.0], [0.5, -0.5]], dtype=float)
    return np.asarray([amp * np.exp(-t / tau) * base for t in times], dtype=float)


def _two_state_basin_network():
    s = np.array([0.0, 1.0])
    F = np.zeros_like(s)
    basins = [
        Basin1D(id=0, minimum=0.0, f_min=0.0, radius=0.0, bounds=np.array([0.0, 0.0])),
        Basin1D(id=1, minimum=1.0, f_min=0.0, radius=0.0, bounds=np.array([1.0, 1.0])),
    ]
    return s, F, BasinNetwork1D(basins=basins, s=s, U=F, labels=np.array([0, 1]))


def test_build_smolu_generator_1d_detailed_balance():
    s = np.linspace(-1.0, 1.0, 41)
    F = 0.5 * s**2 + 0.1 * s
    beta = 0.7
    K = sk.build_smolu_generator_1d(s, F, D=0.03, beta=beta)

    np.testing.assert_allclose(K.sum(axis=1), 0.0, atol=1e-12)
    offdiag = K.copy()
    np.fill_diagonal(offdiag, 0.0)
    assert np.all(offdiag >= 0.0)

    pi = np.exp(-beta * F)
    pi /= pi.sum()
    np.testing.assert_allclose(pi @ K, 0.0, atol=1e-12)


def test_validate_memory_kernel_accepts_and_rejects_helpfully():
    K = _small_generator()
    times = np.linspace(0.0, 0.1, 6)
    Sigma = np.zeros((times.size, 2, 2))

    result = sk.validate_memory_kernel(Sigma, times, K0=K)
    assert result.is_valid
    assert result.Sigma.shape == (6, 2, 2)

    with pytest.raises(ValueError, match="does not estimate memory kernels"):
        sk.validate_memory_kernel(None, times, K0=K)

    with pytest.raises(ValueError, match="ndim=2"):
        sk.validate_memory_kernel(np.zeros((2, 2)), times, K0=K)


def test_enforce_generator_conservation_row_and_column():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])

    row = sk.enforce_generator_conservation(A, convention="row")
    np.testing.assert_allclose(row.sum(axis=1), 0.0, atol=1e-12)

    col = sk.enforce_generator_conservation(A, convention="column")
    np.testing.assert_allclose(col.sum(axis=0), 0.0, atol=1e-12)


def test_zero_memory_matches_markovian_transition_with_small_dt():
    K = _small_generator()
    dt = 1e-4
    n_steps = 80
    times = np.arange(n_steps + 1, dtype=float) * dt
    Sigma = np.zeros((times.size, 2, 2))

    result = sk.propagate_gme_transition_matrix(K, Sigma, times, n_steps=n_steps, dt=dt)
    expected = np.asarray([expm(t * K) for t in result.times])

    np.testing.assert_allclose(result.trajectory, expected, rtol=2e-5, atol=2e-7)


def test_nonzero_exponential_memory_propagates_normalized_probabilities():
    K = _small_generator()
    times = np.linspace(0.0, 0.2, 101)
    Sigma = _memory_stack(times, amp=0.02, tau=0.06)

    result = sk.propagate_gme(
        np.array([1.0, 0.0]),
        K,
        Sigma,
        times,
        n_steps=80,
        dt=times[1] - times[0],
    )

    assert np.all(np.isfinite(result.trajectory))
    np.testing.assert_allclose(result.trajectory.sum(axis=1), 1.0, atol=1e-10)


def test_memory_moments_and_effective_generator_order_zero():
    K = _small_generator()
    times = np.linspace(0.0, 0.2, 21)
    Sigma = _memory_stack(times, amp=0.01, tau=0.05)

    moments = sk.memory_moments(Sigma, times, max_order=1)
    expected_m0 = getattr(np, "trapezoid", np.trapz)(Sigma, times, axis=0)
    np.testing.assert_allclose(moments[0], expected_m0)

    Keff = sk.effective_markov_generator_from_memory(K, Sigma, times, order=0)
    np.testing.assert_allclose(Keff, sk.enforce_generator_conservation(K + expected_m0))


def test_effective_generator_order_one_matches_row_formula():
    K = _small_generator()
    times = np.linspace(0.0, 0.2, 21)
    Sigma = _memory_stack(times, amp=0.01, tau=0.05)
    moments = sk.memory_moments(Sigma, times, max_order=1)

    Keff = sk.effective_markov_generator_from_memory(K, Sigma, times, order=1)
    expected = np.linalg.solve((np.eye(2) + moments[1]).T, (K + moments[0]).T).T
    np.testing.assert_allclose(Keff, expected)


def test_effective_generator_higher_order_returns_diagnostics():
    K = _small_generator()
    times = np.linspace(0.0, 0.2, 21)
    Sigma = np.asarray([0.05 * np.exp(-t / 0.1) * K for t in times])

    result = sk.effective_markov_generator_from_memory(
        K,
        Sigma,
        times,
        order=2,
        return_result=True,
    )

    assert isinstance(result, sk.EffectiveGeneratorResult)
    assert result.diagnostics["converged"]
    assert result.diagnostics["generator_like"]
    np.testing.assert_allclose(result.K_eff.sum(axis=1), 0.0, atol=1e-10)


def test_row_column_conventions_are_transpose_consistent():
    K_row = _small_generator()
    K_col = K_row.T
    times = np.linspace(0.0, 0.1, 51)
    Sigma_row = _memory_stack(times, amp=0.01, tau=0.04)
    Sigma_col = np.transpose(Sigma_row, axes=(0, 2, 1))

    row = sk.propagate_gme_transition_matrix(K_row, Sigma_row, times, n_steps=30)
    col = sk.propagate_gme_transition_matrix(
        K_col,
        Sigma_col,
        times,
        n_steps=30,
        convention="column",
    )

    np.testing.assert_allclose(col.trajectory, np.transpose(row.trajectory, axes=(0, 2, 1)))


def test_run_gme_1d_requires_user_kernel_and_returns_transition_result():
    s = np.linspace(-1.0, 1.0, 31)
    F = 0.25 * (s**2 - 1.0) ** 2

    with pytest.raises(ValueError, match="Please provide Sigma_t"):
        sk.run_gme_1d(s, F, D=0.02)

    times = np.linspace(0.0, 0.01, 11)
    Sigma = np.zeros((times.size, s.size, s.size))
    out = sk.run_gme_1d(
        s,
        F,
        D=0.02,
        beta=1.0,
        Sigma_t=Sigma,
        memory_times=times,
        n_steps=4,
    )

    assert out["experimental"] is True
    assert out["experimental_api"] == "memory-kernel-kinetics/0.1"
    assert out["propagation_kind"] == "transition_matrix"
    assert out["propagation"].trajectory.shape == (5, s.size, s.size)
    assert out["effective_generator"].shape == (s.size, s.size)


def test_grid_generator_coarse_graining_two_state_exact():
    s, F, bn = _two_state_basin_network()
    K_grid = np.array([[-0.3, 0.3], [0.2, -0.2]])

    res = sk.compute_ctmc_generator_from_grid_generator_1d(
        s, F, K_grid, bn, init_weight="uniform", verbose=False
    )

    np.testing.assert_allclose(res["K"], K_grid)
    np.testing.assert_allclose(res["exit_mean"], [1 / 0.3, 1 / 0.2])
    np.testing.assert_allclose(res["p_branch"], [[np.nan, 1.0], [1.0, np.nan]], equal_nan=True)


def test_grid_generator_coarse_graining_multigrid_row_sums():
    s = np.linspace(0.0, 1.0, 101)
    F = 5.0 * (1.0 - (2.0 * s - 1.0) ** 2) ** 2
    F -= F.min()
    beta = 1.0
    K_grid = sk.build_smolu_generator_1d(s, F, D=0.01, beta=beta)
    bn = sk.build_basin_network_from_fes_1d(s, F, verbose=False)

    res = sk.compute_ctmc_generator_from_grid_generator_1d(
        s, F, K_grid, bn, beta=beta, verbose=False
    )

    np.testing.assert_allclose(res["K"].sum(axis=1), 0.0, atol=1e-10)
    row = res["p_branch"].copy()
    np.fill_diagonal(row, np.nan)
    np.testing.assert_allclose(np.nansum(row, axis=1), 1.0, atol=1e-12)


def test_memory_corrected_ctmc_zero_memory_matches_grid_coarse_graining():
    s = np.linspace(0.0, 1.0, 101)
    F = 5.0 * (1.0 - (2.0 * s - 1.0) ** 2) ** 2
    F -= F.min()
    beta = 1.0
    D = 0.01
    times = np.linspace(0.0, 0.1, 11)
    Sigma = np.zeros((times.size, s.size, s.size))

    out = sk.run_memory_corrected_ctmc_1d(
        s,
        F,
        D,
        beta=beta,
        Sigma_t=Sigma,
        memory_times=times,
        memory_order=0,
        core_fraction=None,
        verbose=False,
    )
    bn = sk.build_basin_network_from_fes_1d(s, F, verbose=False)
    K_grid = sk.build_smolu_generator_1d(s, F, D, beta=beta)
    ref = sk.compute_ctmc_generator_from_grid_generator_1d(
        s, F, K_grid, bn, beta=beta, verbose=False
    )

    assert out["experimental"] is True
    assert out["experimental_api"] == "memory-kernel-kinetics/0.1"
    np.testing.assert_allclose(out["K"], ref["K"])


def test_memory_corrected_ctmc_rejects_bad_kernel_shape():
    s = np.linspace(0.0, 1.0, 51)
    F = 5.0 * (1.0 - (2.0 * s - 1.0) ** 2) ** 2
    times = np.linspace(0.0, 0.1, 11)

    with pytest.raises(ValueError, match="Please provide Sigma_t"):
        sk.run_memory_corrected_ctmc_1d(
            s,
            F,
            D=0.01,
            beta=1.0,
            memory_times=times,
            verbose=False,
        )

    with pytest.raises(ValueError, match="K0 has shape"):
        sk.run_memory_corrected_ctmc_1d(
            s,
            F,
            D=0.01,
            beta=1.0,
            Sigma_t=np.zeros((times.size, 2, 2)),
            memory_times=times,
            verbose=False,
        )


def test_memory_corrected_ctmc_conservative_memory_changes_rates():
    s = np.linspace(0.0, 1.0, 101)
    F = 5.0 * (1.0 - (2.0 * s - 1.0) ** 2) ** 2
    F -= F.min()
    beta = 1.0
    D = 0.01
    times = np.linspace(0.0, 0.2, 21)
    K0 = sk.build_smolu_generator_1d(s, F, D, beta=beta)
    Sigma = np.asarray([0.2 * np.exp(-t / 0.05) / 0.05 * K0 for t in times])

    bn = sk.build_basin_network_from_fes_1d(s, F, verbose=False)
    ref = sk.compute_ctmc_generator_from_grid_generator_1d(
        s, F, K0, bn, beta=beta, verbose=False
    )
    out = sk.run_memory_corrected_ctmc_1d(
        s,
        F,
        D,
        beta=beta,
        Sigma_t=Sigma,
        memory_times=times,
        memory_order=0,
        core_fraction=None,
        verbose=False,
    )

    np.testing.assert_allclose(out["K"].sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(out["K_eff_grid"].sum(axis=1), 0.0, atol=1e-12)
    assert not np.allclose(out["K"], ref["K"])
    assert out["memory_diagnostics"]["generator_like"]
