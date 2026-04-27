#!/usr/bin/env python3
"""Example 07 - Fast 1D Langevin MFPT network estimation.

This example compares the original Python trajectory engine with the
``engine="auto"`` path in ``compute_mfpt_network``.  When the optional
compiled backend is built, the auto engine runs the specialized 1D
first-exit loop; otherwise it falls back to the Python implementation.
"""

from pathlib import Path
from time import perf_counter
import os

import matplotlib.pyplot as plt
import numpy as np

import stochkin as sk
from stochkin.fes import FESPotential1D
from stochkin.plotting import _apply_pub_axes
from stochkin.style import publication_style


def _run(label, potential, basin_network, *, engine, n_procs):
    t0 = perf_counter()
    result = sk.compute_mfpt_network(
        potential,
        basin_network,
        dt=0.005,
        max_time=6.0,
        D=0.45,
        beta=1.0,
        bounds=((-1.5, 1.5),),
        boundary="reflect",
        trials_per_basin=500,
        n_procs=n_procs,
        seed=123,
        engine=engine,
    )
    elapsed = perf_counter() - t0
    used = result["params"].get("engine_used", result["method"])
    print(f"{label:>12}: {elapsed:7.3f} s  method={result['method']}  engine={used}")
    print(f"              MFPT matrix:\n{result['mfpt_matrix']}")
    return result, elapsed


def main():
    s = np.linspace(-1.5, 1.5, 401)
    F = 0.25 * (s**2 - 1.0) ** 2
    F -= F.min()

    potential = FESPotential1D(s, F)
    basin_network = sk.build_basin_network_from_fes_1d(
        s,
        F,
        max_basins=2,
        verbose=False,
    )

    print(f"Compiled fast backend available: {sk.fast_langevin1d_backend_available()}")
    python_result, python_time = _run(
        "python",
        potential,
        basin_network,
        engine="python",
        n_procs=1,
    )
    auto_result, auto_time = _run(
        "auto",
        potential,
        basin_network,
        engine="auto",
        n_procs=min(4, os.cpu_count() or 1),
    )

    speedup = python_time / auto_time if auto_time > 0 else np.inf
    print(f"\nSpeedup: {speedup:.2f}x")

    out_path = Path(__file__).with_name("07_fast_langevin_1d.png")
    directions = ["0 to 1", "1 to 0"]
    py_tau = [python_result["mfpt_matrix"][0, 1], python_result["mfpt_matrix"][1, 0]]
    auto_tau = [auto_result["mfpt_matrix"][0, 1], auto_result["mfpt_matrix"][1, 0]]

    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))

        axes[0].plot(s, F, "k-", lw=1.5)
        for basin in basin_network.basins:
            axes[0].axvspan(
                basin.bounds[0],
                basin.bounds[1],
                alpha=0.15,
                label=f"B{basin.id}",
            )
            axes[0].plot(basin.minimum, basin.f_min, "o", ms=5)
        _apply_pub_axes(axes[0], xlabel="x", ylabel="F(x)", title="1D FES")
        axes[0].legend(frameon=False, fontsize=7)

        x = np.arange(len(directions))
        width = 0.36
        axes[1].bar(x - width / 2, py_tau, width, label="python")
        axes[1].bar(x + width / 2, auto_tau, width, label="auto")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(directions)
        _apply_pub_axes(axes[1], ylabel="MFPT", title=f"speedup {speedup:.1f}x")
        axes[1].legend(frameon=False, fontsize=7)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
