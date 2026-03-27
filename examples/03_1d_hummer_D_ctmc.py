#!/usr/bin/env python3
"""Example 03 – 1D PLUMED FES + Hummer D(s) profile → CTMC kinetics
=====================================================================

When short unbiased MD runs have been used to estimate a position-dependent
diffusion coefficient D(s) via Hummer's Bayesian approach, this script shows
how to plug that profile into the CTMC pipeline.

Expected CSV columns (default names):
    x_interface  –  CV positions of bin interfaces
    D_med        –  median D estimate at each interface  [CV² ps⁻¹]

Usage
-----
    python 03_1d_hummer_D_ctmc.py  fes.dat  diffusion_profile.csv  \\
           --crop 4.5 6.5  -T 300

The script writes two output files:
    03_fes_and_D.png     – FES + interpolated D(s) diagnostic
    03_ctmc_rates.png    – Rate matrix heatmap + exit times
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import stochkin as sk

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse():
    p = argparse.ArgumentParser(description="1D FES + Hummer D(s) → CTMC")
    p.add_argument("fes",    help="Path to 1D PLUMED FES file")
    p.add_argument("d_csv",  help="Path to diffusion-profile CSV")
    p.add_argument("--D-xcol", default="x_interface",
                   help="Column name for CV positions in the CSV")
    p.add_argument("--D-col",  default="D_med",
                   help="Column name for D values in the CSV")
    p.add_argument("--D-grid", default="interface",
                   choices=["interface", "center"],
                   help="Whether CSV positions are bin interfaces or centers")
    p.add_argument("--crop",   nargs=2, type=float, metavar=("LO", "HI"))
    p.add_argument("-T",       type=float, default=300.0)
    p.add_argument("--resample-n", type=int, default=500)
    p.add_argument("--out-prefix", default="03")
    return p.parse_args()


def main():
    args = _parse()
    crop = tuple(args.crop) if args.crop else None

    # -----------------------------------------------------------------------
    # Run the full pipeline
    # -----------------------------------------------------------------------
    result = sk.run_1d_ctmc_with_hummer_D(
        fes_path=args.fes,
        d_csv=args.d_csv,
        T=args.T,
        d_xcol=args.D_xcol,
        d_col=args.D_col,
        d_grid=args.D_grid,
        crop=crop,
        resample_n=args.resample_n,
    )

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    K_ns  = result["K_ps"] * 1000
    tau_ns = result["exit_ps"] / 1000
    n = len(result["basin_ids"])
    print(f"\n{n} basins detected")
    print(f"Rate matrix  K  [ns⁻¹]:\n{K_ns}")
    print(f"Exit times  [ns]: {tau_ns}")

    # -----------------------------------------------------------------------
    # Plot 1: FES + D(s)
    # -----------------------------------------------------------------------
    s = result["s"]
    F = result["F"]
    D = np.asarray(result["D_used"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    ax1.plot(s, F, "k-", lw=1.5)
    ax1.set_ylabel("F  [kJ mol⁻¹]")
    ax1.set_title("Free-energy profile")

    ax2.plot(s, D, "b-", lw=1.5)
    ax2.set_ylabel(r"$D(s)$  [CV² ps⁻¹]")
    ax2.set_xlabel("CV")
    ax2.set_title("Hummer D(s) (interpolated)")

    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_fes_and_D.png", dpi=150)
    print(f"Saved {args.out_prefix}_fes_and_D.png")

    # -----------------------------------------------------------------------
    # Plot 2: Rate matrix + exit times
    # -----------------------------------------------------------------------
    K_abs = np.abs(K_ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        Klog = np.where(K_abs > 0, np.log10(K_abs), np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    im = axes[0].imshow(Klog, cmap="magma_r", aspect="auto")
    bids = result["basin_ids"]
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels([f"B{i}" for i in bids])
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels([f"B{i}" for i in bids])
    axes[0].set_title(r"$\log_{10}|K_{ij}|$  [ns⁻¹]")
    plt.colorbar(im, ax=axes[0])

    axes[1].bar(range(n), tau_ns, color="steelblue")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels([f"B{i}" for i in bids])
    axes[1].set_ylabel("Exit time  [ns]")
    axes[1].set_title("Mean exit times")

    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_ctmc_rates.png", dpi=150)
    print(f"Saved {args.out_prefix}_ctmc_rates.png")


if __name__ == "__main__":
    main()
