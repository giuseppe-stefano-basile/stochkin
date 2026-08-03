#!/usr/bin/env python3
"""Example 09 - memory-corrected CTMC basin rates.

This example treats a user-supplied memory kernel as an extension of the
standard 1D CTMC workflow:

1. build the Markovian Smoluchowski grid generator K0 from F(s), D(s);
2. supply Sigma_t on that exact grid;
3. compute a moment-resummed effective grid generator;
4. coarse-grain the corrected grid generator into basin-to-basin CTMC rates.

No memory kernel is estimated here.  The synthetic Sigma_t is constructed only
so the example is fully reproducible.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stochkin as sk
from stochkin.experimental import memory as memory_kinetics
from stochkin.plotting import _apply_pub_axes
from stochkin.style import publication_style


R_GAS = 8.314462618e-3


def make_double_well(n=41):
    """Return a compact synthetic two-basin FES."""

    s = np.linspace(-1.4, 1.4, int(n))
    F = 6.0 * (s * s - 1.0) ** 2
    F -= F.min()
    return s, F


def make_jump_generator(s, left_to_right=1.0e-5, right_to_left=0.5e-5):
    """Small conservative nonlocal correction between the two wells."""

    G = np.zeros((s.size, s.size), dtype=float)
    left = np.argsort(np.abs(s + 1.0))[:3]
    right = np.argsort(np.abs(s - 1.0))[:3]

    # These are intentionally tiny grid rates; basin kinetics are much slower
    # than local grid relaxation, so even small nonlocal memory can be visible.
    for i in left:
        G[i, right] += float(left_to_right) / right.size
    for i in right:
        G[i, left] += float(right_to_left) / left.size

    np.fill_diagonal(G, 0.0)
    for i in range(G.shape[0]):
        G[i, i] = -float(np.sum(G[i, :]))
    return G


def make_user_memory_kernel(K0, G_jump, memory_times):
    """Synthetic user-supplied Sigma_t with units time^-2."""

    t = np.asarray(memory_times, dtype=float)
    local_envelope = 0.30 * np.exp(-t / 0.35) / 0.35
    jump_envelope = (
        (1.0 - np.exp(-t / 0.20))
        * np.exp(-t / 0.90)
        / 0.90
        * (1.0 + 0.30 * np.cos(2.0 * np.pi * t / 0.75))
    )
    Sigma_t = np.asarray(
        [a * K0 + b * G_jump for a, b in zip(local_envelope, jump_envelope)],
        dtype=float,
    )
    components = {
        "local": np.asarray([a * K0 for a in local_envelope], dtype=float),
        "jump": np.asarray([b * G_jump for b in jump_envelope], dtype=float),
    }
    return Sigma_t, components


def print_rates(label, result):
    K = result["K"]
    print(f"{label:>9s}: k(L->R)={K[0, 1]:.6e}, k(R->L)={K[1, 0]:.6e}")
    print(K)


def main():
    s, F = make_double_well()
    D = 1.0e-3

    # Use basin cores for rate extraction.  The grid contains s=0 exactly, so
    # full-basin labels would assign the barrier point to one side and create
    # a small artificial Markovian left/right asymmetry.
    core_fraction = 0.05

    # Synthetic units: choosing this temperature makes beta = 1 for F.
    T = 1.0 / R_GAS
    beta = 1.0
    memory_times = np.linspace(0.0, 2.5, 251)

    K0_grid = memory_kinetics.build_smolu_generator_1d(s, F, D, beta=beta)
    G_jump = make_jump_generator(s)
    Sigma_t, components = make_user_memory_kernel(K0_grid, G_jump, memory_times)

    markov = sk.run_1d_ctmc(
        s,
        F,
        D,
        T=T,
        max_basins=2,
        core_fraction=core_fraction,
        verbose=False,
    )

    corrected = {}
    caught_warnings = {}
    component_results = {}
    for name, Sigma_component in components.items():
        component_results[name] = memory_kinetics.run_memory_corrected_ctmc_1d(
            s,
            F,
            D,
            Sigma_t=Sigma_component,
            memory_times=memory_times,
            memory_order=0,
            T=T,
            max_basins=2,
            core_fraction=core_fraction,
            verbose=False,
        )

    for order in (0, 1, 3):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            corrected[order] = memory_kinetics.run_memory_corrected_ctmc_1d(
                s,
                F,
                D,
                Sigma_t=Sigma_t,
                memory_times=memory_times,
                memory_order=order,
                T=T,
                max_basins=2,
                core_fraction=core_fraction,
                effective_damping=0.05,
                effective_max_iter=1000,
                verbose=False,
            )
        caught_warnings[order] = [str(w.message) for w in caught]

    print("Basin CTMC rate matrices [time^-1]")
    print_rates("Markov", markov)
    print_rates("local", component_results["local"])
    print_rates("nonlocal", component_results["jump"])
    for order, result in corrected.items():
        diag = result["memory_diagnostics"]
        print_rates(f"order {order}", result)
        print(
            "          diagnostics: "
            f"generator_like={diag['generator_like']}, "
            f"converged={diag['converged']}, "
            f"min_offdiag={diag['min_offdiagonal']:.3e}, "
            f"warnings={len(caught_warnings[order])}"
        )

    labels = ["Markov", "local", "nonlocal", "order 0", "order 1", "order 3"]
    plotted_results = [
        markov,
        component_results["local"],
        component_results["jump"],
        corrected[0],
        corrected[1],
        corrected[3],
    ]
    rate_lr = [result["K"][0, 1] for result in plotted_results]
    rate_rl = [result["K"][1, 0] for result in plotted_results]

    memory_norm = np.linalg.norm(Sigma_t.reshape(memory_times.size, -1), axis=1)
    local_norm = np.linalg.norm(
        components["local"].reshape(memory_times.size, -1), axis=1
    )
    jump_norm = np.linalg.norm(
        components["jump"].reshape(memory_times.size, -1), axis=1
    )

    out_path = Path(__file__).with_name("09_memory_corrected_ctmc_rates.png")
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))

        ax = axes[0, 0]
        ax.plot(s, F, "k-", lw=1.5)
        ax.set_xlim(float(s[0]), float(s[-1]))
        _apply_pub_axes(ax, xlabel="s", ylabel="F(s)", title="double-well FES")

        ax = axes[0, 1]
        floor = 1.0e-12
        ax.plot(
            memory_times,
            memory_norm / max(float(memory_norm.max()), floor),
            "k-",
            lw=1.4,
            label="total",
        )
        ax.plot(
            memory_times,
            local_norm / max(float(local_norm.max()), floor),
            "C0--",
            lw=1.2,
            label="local",
        )
        ax.plot(
            memory_times,
            jump_norm / max(float(jump_norm.max()), floor),
            "C3-.",
            lw=1.2,
            label="nonlocal",
        )
        ax.legend(frameon=False, fontsize=7)
        _apply_pub_axes(
            ax,
            xlabel="memory time",
            ylabel="relative component norm",
            title="memory components",
        )

        ax = axes[1, 0]
        x = np.arange(len(labels))
        width = 0.36
        ax.bar(x - width / 2, rate_lr, width, color="C0", label="L->R")
        ax.bar(x + width / 2, rate_rl, width, color="C3", label="R->L")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend(frameon=False, fontsize=7)
        _apply_pub_axes(
            ax,
            xlabel="model",
            ylabel="basin rate",
            title="selected rate constants",
        )

        ax = axes[1, 1]
        ax.axis("off")
        text = [
            "Rate matrices [time^-1]",
            "",
            "Markov:",
            np.array2string(markov["K"], precision=3, suppress_small=False),
            "",
            "Local only:",
            np.array2string(
                component_results["local"]["K"],
                precision=3,
                suppress_small=False,
            ),
            "",
            "Nonlocal only:",
            np.array2string(
                component_results["jump"]["K"],
                precision=3,
                suppress_small=False,
            ),
            "",
            "Full memory order 0:",
            np.array2string(corrected[0]["K"], precision=3, suppress_small=False),
            "",
            "order 3 grid diagnostics:",
            f"generator_like = {corrected[3]['memory_diagnostics']['generator_like']}",
            f"min offdiag = {corrected[3]['memory_diagnostics']['min_offdiagonal']:.2e}",
        ]
        ax.text(0.0, 1.0, "\n".join(text), va="top", family="monospace", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
