#!/usr/bin/env python
"""Example 06 — Uncertainty propagation on a 1-D double-well FES.

Demonstrates how to use :func:`stochkin.bootstrap_ctmc_1d` to estimate
confidence intervals on rates, exit times, and branching probabilities
when the free-energy surface F(s) and the diffusion coefficient D(s)
have known uncertainties.

Two scenarios are shown:

1. **Uniform error bars**: scalar σ_F and relative σ_D.
2. **Position-dependent error bars**: per-grid-point arrays σ_F(s)
   and D_lo(s) / D_hi(s) mimicking Hummer-style posteriors.

Output: a 2×2 figure saved to ``06_uncertainty.png``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import stochkin as sk
from stochkin.style import publication_style
from stochkin.plotting import _apply_pub_axes

# ------------------------------------------------------------------
# Synthetic FES and D(s)
# ------------------------------------------------------------------

def make_synthetic_fes(n: int = 300, barrier: float = 8.0):
    """Two-basin 1-D double well: F(s) = barrier * (1 - (2s-1)^2)^2."""
    s = np.linspace(0.0, 1.0, n)
    x = 2.0 * s - 1.0
    F = barrier * (1.0 - x ** 2) ** 2
    F -= F.min()
    return s, F


def make_synthetic_D(s: np.ndarray, D0: float = 0.02, amp: float = 0.005):
    """Spatially varying D(s) with a dip at the barrier (center)."""
    return D0 - amp * np.exp(-((s - 0.5) ** 2) / (2 * 0.05 ** 2))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-bootstrap", type=int, default=200,
                   help="Number of bootstrap replicates (default 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--outdir", type=str, default=".",
                   help="Output directory")
    return p.parse_args(argv)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    s, F = make_synthetic_fes()
    D = make_synthetic_D(s)

    # ---- Scenario 1: uniform error bars ----
    print("=== Scenario 1: uniform σ_F = 0.5 kJ/mol, 30 % relative D error ===")
    res1 = sk.bootstrap_ctmc_1d(
        s, F, D,
        F_err=0.5,
        D_rel_err=0.3,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        T=300.0,
        time_unit="ps",
        verbose=True,
    )
    print(res1.summary("ps"))
    print()

    # ---- Scenario 2: position-dependent error bars (Hummer-style) ----
    print("=== Scenario 2: position-dependent σ_F(s) and D_lo/D_hi ===")

    # Larger uncertainty at the barrier than at the minima
    F_sigma = 0.3 + 0.8 * np.exp(-((s - 0.5) ** 2) / (2 * 0.08 ** 2))

    # Construct D credible interval (approximately ±1.5 σ on log scale)
    D_lo = D * np.exp(-0.5)
    D_hi = D * np.exp(+0.5)

    res2 = sk.bootstrap_ctmc_1d(
        s, F, D,
        F_err=F_sigma,
        D_lo=D_lo,
        D_hi=D_hi,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
        T=300.0,
        time_unit="ps",
        verbose=True,
    )
    print(res2.summary("ps"))

    # ---- Plot ----
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plot.")
        return

    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5),
                                 gridspec_kw={"hspace": 0.45, "wspace": 0.38})

        # Panel (a): FES with error band
        ax = axes[0, 0]
        ax.fill_between(s, F - F_sigma, F + F_sigma,
                        alpha=0.25, color="C0", label="σ_F(s)")
        ax.plot(s, F, "C0-", lw=1.5, label="F(s)")
        ax.set_xlabel("s")
        ax.set_ylabel("F  [kJ/mol]")
        ax.set_title("(a) FES ± uncertainty")
        ax.legend(fontsize=8)
        _apply_pub_axes(ax)

        # Panel (b): D with CI band
        ax = axes[0, 1]
        ax.fill_between(s, D_lo, D_hi, alpha=0.25, color="C1",
                        label="D CI")
        ax.plot(s, D, "C1-", lw=1.5, label="D(s)")
        ax.set_xlabel("s")
        ax.set_ylabel("D  [CV²/ps]")
        ax.set_title("(b) Diffusion ± CI")
        ax.legend(fontsize=8)
        _apply_pub_axes(ax)

        # Panel (c): rate distribution (scenario 2, off-diagonal K[0,1])
        ax = axes[1, 0]
        k01 = res2.K_samples[:, 0, 1]
        ax.hist(k01, bins=30, density=True, alpha=0.7, color="C2",
                edgecolor="white", linewidth=0.5)
        ax.axvline(res2.K_mean[0, 1], color="k", ls="--", lw=1.2,
                   label=f"mean = {res2.K_mean[0, 1]:.4g}")
        ax.axvline(res2.K_ci_lo[0, 1], color="grey", ls=":", lw=1)
        ax.axvline(res2.K_ci_hi[0, 1], color="grey", ls=":", lw=1,
                   label=f"{res2.confidence_level:.0%} CI")
        ax.set_xlabel("k₀→₁  [ps⁻¹]")
        ax.set_ylabel("density")
        ax.set_title("(c) Rate 0→1 bootstrap")
        ax.legend(fontsize=8)
        _apply_pub_axes(ax)

        # Panel (d): exit-time distribution (scenario 2, basin 0)
        ax = axes[1, 1]
        tau0 = res2.exit_mean_samples[:, 0]
        ax.hist(tau0, bins=30, density=True, alpha=0.7, color="C3",
                edgecolor="white", linewidth=0.5)
        ax.axvline(res2.exit_mean_mean[0], color="k", ls="--", lw=1.2,
                   label=f"mean = {res2.exit_mean_mean[0]:.4g}")
        ax.axvline(res2.exit_mean_ci_lo[0], color="grey", ls=":", lw=1)
        ax.axvline(res2.exit_mean_ci_hi[0], color="grey", ls=":", lw=1,
                   label=f"{res2.confidence_level:.0%} CI")
        ax.set_xlabel("⟨τ_exit⟩  [ps]")
        ax.set_ylabel("density")
        ax.set_title("(d) Exit time basin 0 bootstrap")
        ax.legend(fontsize=8)
        _apply_pub_axes(ax)

        fig.savefig(outdir / "06_uncertainty.png", dpi=300,
                    bbox_inches="tight")
        print(f"\nFigure saved to {outdir / '06_uncertainty.png'}")
        plt.close(fig)


if __name__ == "__main__":
    main()
