import numpy as np
import pytest

import stochkin as sk
from stochkin.fes import FESPotential1D
from stochkin.potentials import Basin1D, BasinNetwork1D, build_basin_network_from_potential
from stochkin.potentials import simple_double_well_2d


def _flat_two_basin_1d():
    s = np.linspace(-1.0, 1.0, 201)
    U = np.zeros_like(s)
    labels = np.where(s < 0.0, 0, 1).astype(int)

    left = labels == 0
    right = labels == 1
    basins = [
        Basin1D(
            id=0,
            minimum=-0.5,
            f_min=0.0,
            radius=0.5,
            bounds=np.array([s[left].min(), s[left].max()]),
        ),
        Basin1D(
            id=1,
            minimum=0.5,
            f_min=0.0,
            radius=0.5,
            bounds=np.array([s[right].min(), s[right].max()]),
        ),
    ]
    return FESPotential1D(s, U), BasinNetwork1D(basins=basins, s=s, U=U, labels=labels)


def test_auto_engine_falls_back_for_unsupported_2d_network():
    basin_network = build_basin_network_from_potential(
        simple_double_well_2d,
        xlim=(-1.5, 1.5),
        ylim=(-0.8, 0.8),
        nx=41,
        ny=31,
        max_basins=2,
        verbose=False,
    )

    result = sk.compute_mfpt_network(
        simple_double_well_2d,
        basin_network,
        dt=0.01,
        max_time=0.01,
        D=0.2,
        beta=1.0,
        bounds=((-1.5, 1.5), (-0.8, 0.8)),
        trials_per_basin=1,
        n_procs=1,
        seed=1,
        engine="auto",
    )

    assert result["method"] == "traj_first_exit"
    assert result["params"]["engine_used"] == "python"
    assert "BasinNetwork1D" in result["params"]["fast_fallback_reason"]


@pytest.mark.skipif(
    not sk.fast_langevin1d_backend_available(),
    reason="compiled fast Langevin backend is not built",
)
def test_fast_1d_engine_preserves_network_schema_and_counts():
    potential, basin_network = _flat_two_basin_1d()

    result = sk.compute_mfpt_network(
        potential,
        basin_network,
        dt=0.005,
        max_time=4.0,
        D=0.5,
        beta=1.0,
        bounds=((-1.0, 1.0),),
        trials_per_basin=80,
        n_procs=2,
        seed=7,
        engine="fast",
    )

    for key in [
        "mfpt",
        "mfpt_matrix",
        "std_matrix",
        "n_samples",
        "transition_times",
        "first_exit_times",
        "exit_to_counts",
        "success_counts",
        "censored_counts",
        "attempts_per_basin",
        "params",
        "method",
    ]:
        assert key in result

    assert result["method"] == "traj_first_exit_fast_1d"
    assert result["params"]["engine_used"] == "fast_1d_compiled"
    attempts = result["attempts_per_basin"]
    accounted = result["success_counts"].sum(axis=1) + result["censored_counts"]
    np.testing.assert_array_equal(accounted, attempts)


@pytest.mark.skipif(
    not sk.fast_langevin1d_backend_available(),
    reason="compiled fast Langevin backend is not built",
)
def test_fast_1d_engine_statistically_matches_python_reference():
    potential, basin_network = _flat_two_basin_1d()

    common = dict(
        dt=0.005,
        max_time=6.0,
        D=0.5,
        beta=1.0,
        bounds=((-1.0, 1.0),),
        trials_per_basin=300,
        seed=11,
    )
    python_result = sk.compute_mfpt_network(
        potential,
        basin_network,
        n_procs=1,
        engine="python",
        **common,
    )
    fast_result = sk.compute_mfpt_network(
        potential,
        basin_network,
        n_procs=2,
        engine="fast",
        **common,
    )

    for i, j in [(0, 1), (1, 0)]:
        tau_python = float(python_result["mfpt_matrix"][i, j])
        tau_fast = float(fast_result["mfpt_matrix"][i, j])
        assert np.isfinite(tau_python)
        assert np.isfinite(tau_fast)
        assert abs(tau_fast - tau_python) / tau_python < 0.35
