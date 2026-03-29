"""Smoke tests for stochkin.
All tests use analytic potentials only – no external FES files required.
"""
import numpy as np
import pytest
import stochkin as sk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _simple_1d_fes(n=200, barrier=5.0):
    """Two-basin 1D FES: F(s) = barrier * (1 - (2s/L - 1)^2)^2  on [0, 1]."""
    s = np.linspace(0.0, 1.0, n)
    # saddle at centre, minima at edges
    x = 2.0 * s - 1.0          # x ∈ [-1, 1]
    F = barrier * (1.0 - x**2) ** 2
    F -= F.min()
    return s, F


# ---------------------------------------------------------------------------
# Package-level
# ---------------------------------------------------------------------------
def test_version():
    assert hasattr(sk, "__version__")
    assert sk.__version__.count(".") >= 1


def test_public_api():
    for name in [
        "run_1d_ctmc",
        "run_1d_ctmc_from_plumed",
        "run_1d_ctmc_with_hummer_D",
        "run_mfep_ctmc",
        "interface_to_centers",
        "interpolate_D_to_grid",
        "build_basin_network_from_fes_1d",
        "compute_ctmc_generator_fpe_1d",
        "load_plumed_fes_1d",
        "load_plumed_fes_2d",
    ]:
        assert hasattr(sk, name), f"Missing public symbol: {name}"


# ---------------------------------------------------------------------------
# Basin detection
# ---------------------------------------------------------------------------
def test_basin_detection_1d_two_basins():
    s, F = _simple_1d_fes()
    bn = sk.build_basin_network_from_fes_1d(s, F, verbose=False)
    assert bn.n_basins == 2, f"Expected 2 basins, got {bn.n_basins}"
    assert len(bn.labels) == len(s)


def test_basin_detection_1d_max_basins():
    s, F = _simple_1d_fes()
    bn = sk.build_basin_network_from_fes_1d(s, F, max_basins=1, verbose=False)
    assert bn.n_basins == 1


# ---------------------------------------------------------------------------
# workflows.run_1d_ctmc  (core function)
# ---------------------------------------------------------------------------
def test_run_1d_ctmc_returns_dict():
    s, F = _simple_1d_fes()
    res = sk.run_1d_ctmc(s, F, D=0.01, T=300.0, verbose=False)
    for key in ["s", "F", "D_used", "kT", "K", "K_ps",
                "exit_mean", "exit_ps", "k_out", "k_out_ps",
                "p_branch", "labels_full", "basin_ids"]:
        assert key in res, f"Missing key '{key}' in result"


def test_run_1d_ctmc_rate_matrix_shape():
    s, F = _simple_1d_fes()
    res = sk.run_1d_ctmc(s, F, D=0.01, T=300.0, verbose=False)
    n = len(res["basin_ids"])
    assert res["K"].shape == (n, n)
    assert res["K_ps"].shape == (n, n)


def test_run_1d_ctmc_diagonal_negative():
    """Diagonal of a valid rate matrix must be non-positive."""
    s, F = _simple_1d_fes()
    res = sk.run_1d_ctmc(s, F, D=0.01, T=300.0, verbose=False)
    diag = np.diag(res["K"])
    assert np.all(diag <= 0.0), f"Positive diagonal: {diag}"


def test_run_1d_ctmc_row_sum_zero():
    """Off-diagonal contributions: rows of K must sum to ~0."""
    s, F = _simple_1d_fes()
    res = sk.run_1d_ctmc(s, F, D=0.01, T=300.0, verbose=False)
    row_sums = res["K"].sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-8,
                               err_msg="Row sums of K are not zero")


def test_run_1d_ctmc_exit_times_positive():
    s, F = _simple_1d_fes()
    res = sk.run_1d_ctmc(s, F, D=0.01, T=300.0, verbose=False)
    assert np.all(res["exit_mean"] > 0), "Non-positive exit times"


def test_run_1d_ctmc_time_unit_ns():
    """Results in ns should be consistent with ps (factor 1000)."""
    s, F = _simple_1d_fes()
    D_ps = 0.01   # CV² ps⁻¹
    D_ns = D_ps * 1000.0  # CV² ns⁻¹

    res_ps = sk.run_1d_ctmc(s, F, D=D_ps, T=300.0, time_unit="ps", verbose=False)
    res_ns = sk.run_1d_ctmc(s, F, D=D_ns, T=300.0, time_unit="ns", verbose=False)

    # K_ps should be equal regardless of time_unit
    np.testing.assert_allclose(
        res_ps["K_ps"], res_ns["K_ps"], rtol=1e-6,
        err_msg="K_ps differs between ps and ns inputs"
    )


def test_run_1d_ctmc_variable_D():
    """Variable D(s) should work without errors."""
    s, F = _simple_1d_fes()
    D = 0.01 + 0.005 * np.sin(np.pi * s)  # spatially varying
    res = sk.run_1d_ctmc(s, F, D=D, T=300.0, verbose=False)
    assert res["K"].shape[0] == len(res["basin_ids"])


def test_run_1d_ctmc_no_core():
    """core_fraction=None should run without error."""
    s, F = _simple_1d_fes()
    res = sk.run_1d_ctmc(s, F, D=0.01, T=300.0, core_fraction=None, verbose=False)
    assert "K" in res


# ---------------------------------------------------------------------------
# workflows.interface_to_centers
# ---------------------------------------------------------------------------
def test_interface_to_centers_shape():
    x = np.linspace(1.0, 5.0, 10)
    D = np.ones(10) * 0.05
    xc, Dc, edges = sk.interface_to_centers(x, D)
    assert xc.size == 11
    assert Dc.size == 11
    assert edges.size == 12


def test_interface_to_centers_constant_D():
    """Harmonic mean of equal values is the same value."""
    x = np.linspace(1.0, 5.0, 10)
    D = np.full(10, 0.05)
    _, Dc, _ = sk.interface_to_centers(x, D, method="harmonic")
    np.testing.assert_allclose(Dc, 0.05, rtol=1e-10)


# ---------------------------------------------------------------------------
# workflows.interpolate_D_to_grid
# ---------------------------------------------------------------------------
def test_interpolate_D_to_grid_length():
    s = np.linspace(0.0, 1.0, 300)
    xD = np.linspace(0.1, 0.9, 20)
    D  = np.full(20, 0.03)
    Dg = sk.interpolate_D_to_grid(s, xD, D)
    assert Dg.size == 300


def test_interpolate_D_to_grid_positive():
    s  = np.linspace(0.0, 1.0, 300)
    xD = np.linspace(0.1, 0.9, 20)
    D  = np.abs(np.random.default_rng(7).standard_normal(20)) * 0.02 + 0.01
    Dg = sk.interpolate_D_to_grid(s, xD, D)
    assert np.all(Dg > 0), "Interpolated D contains non-positive values"


# ---------------------------------------------------------------------------
# Uncertainty propagation
# ---------------------------------------------------------------------------
def test_bootstrap_ctmc_1d_returns_result():
    """bootstrap_ctmc_1d should return an UncertaintyResult."""
    s, F = _simple_1d_fes()
    res = sk.bootstrap_ctmc_1d(
        s, F, D=0.01, F_err=0.3, n_bootstrap=10, seed=0, verbose=False,
    )
    assert isinstance(res, sk.UncertaintyResult)
    assert res.n_bootstrap + res.n_failed == 10


def test_bootstrap_ctmc_1d_shapes():
    s, F = _simple_1d_fes()
    res = sk.bootstrap_ctmc_1d(
        s, F, D=0.01, F_err=0.3, D_rel_err=0.2,
        n_bootstrap=15, seed=1, verbose=False,
    )
    nb = res.reference["K"].shape[0]
    assert res.K_mean.shape == (nb, nb)
    assert res.K_std.shape == (nb, nb)
    assert res.K_ci_lo.shape == (nb, nb)
    assert res.K_ci_hi.shape == (nb, nb)
    assert res.K_samples.shape[1:] == (nb, nb)
    assert res.exit_mean_mean.shape == (nb,)
    assert res.k_out_samples.shape[1] == nb


def test_bootstrap_ctmc_1d_ci_contains_reference():
    """The reference (unperturbed) rate should lie within the wide CI."""
    s, F = _simple_1d_fes()
    res = sk.bootstrap_ctmc_1d(
        s, F, D=0.01, F_err=0.3, n_bootstrap=30, seed=2, verbose=False,
    )
    K_ref = res.reference["K"]
    # At least one off-diagonal element should be within CI
    nb = K_ref.shape[0]
    inside = 0
    for i in range(nb):
        for j in range(nb):
            if i == j:
                continue
            if res.K_ci_lo[i, j] <= K_ref[i, j] <= res.K_ci_hi[i, j]:
                inside += 1
    assert inside > 0, "Reference rate not inside any CI"


def test_bootstrap_ctmc_1d_zero_error():
    """Zero error bars should produce zero spread."""
    s, F = _simple_1d_fes()
    res = sk.bootstrap_ctmc_1d(
        s, F, D=0.01, F_err=0.0, D_err=0.0,
        n_bootstrap=10, seed=3, verbose=False,
    )
    # All samples should be identical to the reference
    np.testing.assert_allclose(res.K_std, 0.0, atol=1e-10)


def test_bootstrap_ctmc_1d_D_lo_hi():
    """Passing D_lo/D_hi (log-normal perturbation) should work."""
    s, F = _simple_1d_fes()
    D = np.full_like(s, 0.01)
    D_lo = D * 0.5
    D_hi = D * 2.0
    res = sk.bootstrap_ctmc_1d(
        s, F, D=D, D_lo=D_lo, D_hi=D_hi,
        n_bootstrap=10, seed=4, verbose=False,
    )
    assert res.n_bootstrap > 0
    assert res.K_std.max() > 0  # non-trivial spread


def test_bootstrap_ctmc_1d_summary_string():
    s, F = _simple_1d_fes()
    res = sk.bootstrap_ctmc_1d(
        s, F, D=0.01, F_err=0.3, n_bootstrap=10, seed=5, verbose=False,
    )
    txt = res.summary("ps")
    assert "Bootstrap" in txt
    assert "ps" in txt


def test_uncertainty_in_public_api():
    for name in ["bootstrap_ctmc_1d", "bootstrap_ctmc_with_hummer_D",
                 "UncertaintyResult"]:
        assert hasattr(sk, name), f"Missing public symbol: {name}"
