#!/usr/bin/env python3
"""Example 04 – 2D PLUMED FES → multi-MFEP → all-basin CTMC kinetics
======================================================================

This script demonstrates the full multi-basin MFEP workflow:

  1. Load a 2D PLUMED FES (e.g., from well-tempered metadynamics).
  2. Auto-detect **all** basins on the 2D surface.
  3. For every pair of basins, compute the MFEP and extract the
     pairwise forward/backward rates from the 1D arc-length CTMC.
  4. Assemble the full N×N rate matrix and print exit times.

Usage
-----
    # Using the bundled synthetic 3-basin data:
    python 04_mfep_ctmc.py  data/synthetic_2d_fes.dat  --D-s 0.05  -T 300

    # Or with your own 2D PLUMED FES:
    python 04_mfep_ctmc.py  fes_2d.dat  --D-s 0.04  -T 300

Output
------
    04_basins_on_fes.png   – detected basins overlaid on the 2D FES
    04_rate_matrix.png     – full N×N rate matrix heatmap
    04_exit_times.png      – bar chart of mean exit times
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import stochkin as sk
from stochkin.style import publication_style, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE
from stochkin.plotting import _apply_pub_axes, _apply_pub_cbar

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse():
    p = argparse.ArgumentParser(
        description="2D FES → auto-detect basins → pairwise MFEPs → CTMC")
    p.add_argument("fes2d",  help="Path to 2D PLUMED FES file")
    p.add_argument("--D-s",   type=float, default=0.04,
                   help="Diffusion along arc-length  [arc-length² ps⁻¹]")
    p.add_argument("-T",      type=float, default=300.0,
                   help="Temperature in K")
    p.add_argument("--neb-images", type=int, default=120)
    p.add_argument("--neb-steps",  type=int, default=8000)
    p.add_argument("--max-basins", type=int, default=None,
                   help="Keep only the N deepest basins (default: all)")
    p.add_argument("--core-fraction", type=float, default=0.05,
                   help="Core fraction for CTMC entry/exit  (default 0.05)")
    p.add_argument("--out-prefix", default="04")
    return p.parse_args()


def main():
    args = _parse()

    # ------------------------------------------------------------------
    # Multi-MFEP CTMC
    # ------------------------------------------------------------------
    print(f"Loading 2D FES: {args.fes2d}")
    result = sk.run_multi_mfep_ctmc(
        fes2d_path=args.fes2d,
        D_s=args.D_s,
        T=args.T,
        neb_images=args.neb_images,
        neb_steps=args.neb_steps,
        max_basins=args.max_basins,
        core_fraction=args.core_fraction,
    )

    basin_net = result["basin_network"]
    K_ns      = result["K_ps"] * 1000   # ps⁻¹ → ns⁻¹
    tau_ns    = result["exit_ps"] / 1000
    n         = len(result["basin_ids"])

    print(f"\n{'='*60}")
    print(f"{n} basins detected")
    print(f"Rate matrix  K  [ns⁻¹]:\n{K_ns}")
    print(f"Exit times  [ns]: {tau_ns}")

    # ------------------------------------------------------------------
    # Plot 1: 2D FES + detected basins (publication style)
    # ------------------------------------------------------------------
    x_grid, y_grid, fes_grid = sk.load_plumed_fes_2d(args.fes2d, verbose=False)
    kT = result["kT"]

    with publication_style():
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        cs = ax.contourf(
            x_grid, y_grid, (fes_grid / kT).T,
            levels=np.linspace(0, 15, 30), cmap="rainbow_r",
        )
        cbar = fig.colorbar(cs, ax=ax)
        _apply_pub_cbar(cbar, label=r"$F / k_\mathrm{B}T$")

        # Plot basin minima and MFEP legs
        colors = plt.get_cmap("tab10")
        for b in basin_net.basins:
            ax.plot(*b.minimum, "o", color="white", ms=10, zorder=5)
            ax.text(b.minimum[0], b.minimum[1], f" B{b.id}",
                    color="white", va="center", fontsize=9, fontweight="bold",
                    zorder=6)

        for (i, j), leg in result["legs"].items():
            path = leg["mfep_path"]
            ax.plot(path.x, path.y, "--", color="white", lw=1.5, alpha=0.8)

        _apply_pub_axes(ax, r"CV$_1$", r"CV$_2$", "Basins + MFEPs on 2D FES")
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_basins_on_fes.png", dpi=300)
    print(f"Saved {args.out_prefix}_basins_on_fes.png")

    # ------------------------------------------------------------------
    # Plot 2: Full rate matrix heatmap (publication style)
    # ------------------------------------------------------------------
    K_abs = np.abs(K_ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        Klog = np.where(K_abs > 0, np.log10(K_abs), np.nan)

    with publication_style():
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        im = ax.imshow(Klog, cmap="magma_r", aspect="auto")
        bids = result["basin_ids"]
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"B{i}" for i in bids])
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"B{i}" for i in bids])
        cbar = fig.colorbar(im, ax=ax)
        _apply_pub_cbar(cbar, label=r"$\log_{10}|K_{ij}|$  [ns$^{-1}$]")
        _apply_pub_axes(ax, title=f"Rate matrix ({n} basins)")
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_rate_matrix.png", dpi=300)
    print(f"Saved {args.out_prefix}_rate_matrix.png")

    # ------------------------------------------------------------------
    # Plot 3: Exit times bar chart (publication style)
    # ------------------------------------------------------------------
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        bars = ax.bar(range(n), tau_ns, color="steelblue")
        ax.set_yscale("log")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"B{i}" for i in bids])
        for bar, t in zip(bars, tau_ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{t:.1f}", ha="center", va="bottom", fontsize=8)
        _apply_pub_axes(ax, xlabel="Basin", ylabel="Exit time  [ns]",
                        title="Mean exit times")
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_exit_times.png", dpi=300)
    print(f"Saved {args.out_prefix}_exit_times.png")


if __name__ == "__main__":
    main()
