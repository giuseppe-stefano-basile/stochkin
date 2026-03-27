#!/usr/bin/env python3
"""Example 02 – 1D PLUMED FES + constant D → CTMC kinetics
============================================================

This script shows the simplest possible workflow:
  1. Load a 1D FES file produced by PLUMED metadynamics.
  2. Run the full CTMC pipeline with a constant diffusion coefficient.
  3. Print the rate matrix and exit times.
  4. Save a summary plot.

Usage
-----
    python 02_1d_plumed_fes_ctmc.py fes.dat --D 0.04 --crop 4.5 6.5 -T 300

Adapt the argument defaults below to your system.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import stochkin as sk

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse():
    p = argparse.ArgumentParser(description="1D PLUMED FES → CTMC kinetics")
    p.add_argument("fes",        help="Path to 1D PLUMED FES file")
    p.add_argument("--D",        type=float, default=0.04,
                   help="Constant diffusion coefficient  [CV² ps⁻¹]")
    p.add_argument("--crop",     nargs=2, type=float, metavar=("LO", "HI"),
                   help="Crop the FES to this CV range")
    p.add_argument("-T",         type=float, default=300.0,
                   help="Temperature in K  (default 300)")
    p.add_argument("--resample-n", type=int, default=500,
                   help="Resample FES to N uniform points  (default 500)")
    p.add_argument("--core-fraction", type=float, default=0.05,
                   help="Core fraction for CTMC entry/exit  (default 0.05)")
    p.add_argument("--out",      default="02_ctmc_rates.png",
                   help="Output PNG path")
    return p.parse_args()


def main():
    args = _parse()
    crop = tuple(args.crop) if args.crop else None

    # -----------------------------------------------------------------------
    # Run the CTMC pipeline
    # -----------------------------------------------------------------------
    print(f"Loading FES: {args.fes}")
    result = sk.run_1d_ctmc_from_plumed(
        fes_path=args.fes,
        D=args.D,
        T=args.T,
        crop=crop,
        resample_n=args.resample_n,
        core_fraction=args.core_fraction,
    )

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    K_ns = result["K_ps"] * 1000  # convert ps⁻¹ → ns⁻¹
    tau_ns = result["exit_ps"] / 1000

    print(f"\nBasin ids: {result['basin_ids']}")
    print(f"Rate matrix  K  [ns⁻¹]:\n{K_ns}")
    print(f"\nExit times τ  [ns]:  {tau_ns}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    s      = result["s"]
    F      = result["F"]
    labels = result["labels_full"]

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    # Panel (a): FES + basins
    ax = axes[0]
    ax.plot(s, F, "k-", lw=1.5)
    for bid in result["basin_ids"]:
        mask = labels == bid
        ax.fill_between(s, F, where=mask, alpha=0.25,
                        color=cmap(bid), label=f"B{bid}")
    ax.set_xlabel("CV")
    ax.set_ylabel("F  [kJ mol⁻¹]")
    ax.set_title("FES + basins")
    ax.legend(fontsize=8)

    # Panel (b): Rate matrix heatmap
    ax = axes[1]
    K_abs = np.abs(K_ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        Klog = np.where(K_abs > 0, np.log10(K_abs), np.nan)
    im = ax.imshow(Klog, cmap="magma_r", aspect="auto")
    n = K_ns.shape[0]
    ax.set_xticks(range(n)); ax.set_xticklabels([f"B{i}" for i in result["basin_ids"]])
    ax.set_yticks(range(n)); ax.set_yticklabels([f"B{i}" for i in result["basin_ids"]])
    ax.set_title(r"$\log_{10}|K_{ij}|$  [ns⁻¹]")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
