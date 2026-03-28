#!/usr/bin/env python3
"""Example 04 – 2D PLUMED FES → MFEP → 1D arc-length CTMC
===========================================================

This script demonstrates the MFEP-based kinetics workflow:

  1. Load a 2D PLUMED FES (e.g., from well-tempered metadynamics).
  2. Find the minimum free-energy path (MFEP) between two basins using
     Dijkstra + NEB refinement.
  3. Extract the 1D free-energy profile along the arc-length coordinate.
  4. Run the CTMC pipeline on that 1D profile with a constant D_s.

Usage
-----
    # Using the bundled synthetic data:
    python 04_mfep_ctmc.py  data/synthetic_2d_fes.dat  \
           --start 1.5 1.5  --end 6.0 1.5  --D-s 0.05  -T 300

    # Or with your own 2D PLUMED FES:
    python 04_mfep_ctmc.py  fes_2d.dat  \\
           --start 1.0 0.5  --end 5.0 0.5  --D-s 0.04  -T 300

Output
------
    04_mfep_on_fes.png     – MFEP overlaid on the 2D FES
    04_mfep_1d_profile.png – F(s) along the arc-length + detected basins
    04_ctmc_rates.png      – Rate matrix heatmap
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
    p = argparse.ArgumentParser(description="2D FES → MFEP → 1D CTMC")
    p.add_argument("fes2d",  help="Path to 2D PLUMED FES file")
    p.add_argument("--start", nargs=2, type=float, required=True,
                   metavar=("X", "Y"), help="MFEP start point in CV space")
    p.add_argument("--end",   nargs=2, type=float, required=True,
                   metavar=("X", "Y"), help="MFEP end point in CV space")
    p.add_argument("--D-s",   type=float, default=0.04,
                   help="Diffusion along arc-length  [arc-length² ps⁻¹]")
    p.add_argument("-T",      type=float, default=300.0,
                   help="Temperature in K")
    p.add_argument("--neb-images", type=int, default=120)
    p.add_argument("--neb-steps",  type=int, default=3000)
    p.add_argument("--out-prefix", default="04")
    return p.parse_args()


def main():
    args = _parse()

    # ------------------------------------------------------------------
    # MFEP + CTMC
    # ------------------------------------------------------------------
    print(f"Loading 2D FES: {args.fes2d}")
    result = sk.run_mfep_ctmc(
        fes2d_path=args.fes2d,
        start=tuple(args.start),
        end=tuple(args.end),
        D_s=args.D_s,
        T=args.T,
        neb_images=args.neb_images,
        neb_steps=args.neb_steps,
    )

    path   = result["mfep_path"]   # MFEPPath with .x, .y, .s, .F
    K_ns   = result["K_ps"] * 1000
    tau_ns = result["exit_ps"] / 1000
    n      = len(result["basin_ids"])

    print(f"\n{n} basins detected along the MFEP")
    print(f"Rate matrix  K  [ns⁻¹]:\n{K_ns}")
    print(f"Exit times  [ns]: {tau_ns}")

    # ------------------------------------------------------------------
    # Plot 1: 2D FES + MFEP (publication style)
    # ------------------------------------------------------------------
    x_grid, y_grid, fes_grid = sk.load_plumed_fes_2d(args.fes2d, verbose=False)
    kT = result["kT"]

    with publication_style():
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        cs = ax.contourf(
            x_grid, y_grid, (fes_grid / kT).T,
            levels=np.linspace(0, 15, 30), cmap="viridis_r",
        )
        cbar = fig.colorbar(cs, ax=ax)
        _apply_pub_cbar(cbar, label=r"$F / k_\mathrm{B}T$")
        ax.plot(path.x, path.y, "w-", lw=2, label="MFEP")
        ax.plot(*args.start, "ws", ms=10)
        ax.plot(*args.end,   "w^", ms=10)
        _apply_pub_axes(ax, r"CV$_1$", r"CV$_2$", "2D FES + MFEP")
        ax.legend(fontsize=LEGEND_SIZE, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_mfep_on_fes.png", dpi=300)
    print(f"Saved {args.out_prefix}_mfep_on_fes.png")

    # ------------------------------------------------------------------
    # Plot 2: 1D arc-length profile (publication style)
    # ------------------------------------------------------------------
    s      = result["s"]
    F      = result["F"]
    labels = result["labels_full"]

    with publication_style():
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        ax.plot(s, F, "k-", lw=1.5)
        for bid in result["basin_ids"]:
            mask = labels == bid
            ax.fill_between(s, F, where=mask, alpha=0.25,
                            color=cmap(bid), label=f"B{bid}")
        _apply_pub_axes(ax, "Arc-length  s", r"F  [kJ mol$^{-1}$]",
                        "1D profile along MFEP")
        ax.legend(fontsize=LEGEND_SIZE, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_mfep_1d_profile.png", dpi=300)
    print(f"Saved {args.out_prefix}_mfep_1d_profile.png")

    # ------------------------------------------------------------------
    # Plot 3: Rate matrix (publication style)
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
        _apply_pub_axes(ax, title="Rate matrix")
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_ctmc_rates.png", dpi=300)
    print(f"Saved {args.out_prefix}_ctmc_rates.png")


if __name__ == "__main__":
    main()
