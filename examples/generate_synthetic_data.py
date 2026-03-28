#!/usr/bin/env python3
"""Generate synthetic FES and diffusion-profile data for the stochkin examples.

The landscapes are purely illustrative — a "conformational" reaction coordinate
for a fictional toy molecule — and bear no relation to the real ion-solvation
systems studied in the paper.

Output files (written to ``examples/data/``):

  synthetic_1d_fes.dat
      1D triple-well free-energy surface in PLUMED ``sum_hills`` format.
      CV range 0–10 (generic "reaction coordinate  ξ"), 501 grid points.

  synthetic_diffusion_profile.csv
      Position-dependent diffusion coefficient D(ξ) in Hummer format
      (interface positions).  Columns: x_interface, D_med, D_lo, D_hi.

  synthetic_2d_fes.dat
      2D three-basin landscape in PLUMED ``sum_hills`` format.
      CV₁ ∈ [0, 8],  CV₂ ∈ [0, 6],  201 × 151 grid.

Run
---
    python generate_synthetic_data.py          # writes to examples/data/
    python generate_synthetic_data.py --plot   # also saves diagnostic PNGs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)


# ===================================================================
# 1.  Synthetic 1D FES — triple-well
# ===================================================================
def _synthetic_1d_fes(s: np.ndarray) -> np.ndarray:
    """Triple-well along a generic reaction coordinate ξ ∈ [0, 10].

    Three minima at roughly ξ ≈ 1.8, 5.0, 8.3 separated by barriers of
    ~12 and ~8 kJ/mol.  The global minimum is the central well.

    Construction: quadratic envelope + three Gaussian wells.
    """
    envelope = 0.20 * (s - 5.0) ** 2
    well_A = 15.0 * np.exp(-((s - 1.8) ** 2) / 1.2)
    well_B = 22.0 * np.exp(-((s - 5.0) ** 2) / 1.6)
    well_C = 12.0 * np.exp(-((s - 8.3) ** 2) / 0.9)
    F = envelope - well_A - well_B - well_C
    F -= F.min()
    return F


def write_1d_fes(path: Path, n: int = 501) -> None:
    s = np.linspace(0.0, 10.0, n)
    F = _synthetic_1d_fes(s)
    dF = np.gradient(F, s)

    with open(path, "w") as fh:
        fh.write("#! FIELDS xi file.free der_xi\n")
        fh.write(f"#! SET min_xi {s[0]:.1f}\n")
        fh.write(f"#! SET max_xi {s[-1]:.1f}\n")
        fh.write(f"#! SET nbins_xi  {n}\n")
        fh.write("#! SET periodic_xi false\n")
        for i in range(n):
            fh.write(f"   {s[i]:.9f}  {F[i]:.9f}   {dF[i]:.9f}\n")
    print(f"  Wrote {path}  ({n} points)")


# ===================================================================
# 2.  Synthetic diffusion profile D(ξ)
# ===================================================================
def _synthetic_D(x: np.ndarray) -> np.ndarray:
    """Position-dependent D(ξ) — smooth and positive everywhere.

    Base value 0.05 CV² ps⁻¹, with a broad dip near the central barrier
    (diffusion slows where the barrier is high) and a slight rise in the
    right-hand well.
    """
    D0 = 0.050
    dip = 0.035 * np.exp(-((x - 3.4) ** 2) / 0.8)  # barrier dip
    rise = 0.020 * np.exp(-((x - 8.0) ** 2) / 1.5)  # shallow rise
    return D0 - dip + rise


def write_diffusion_csv(path: Path, n_interfaces: int = 42) -> None:
    """Write Hummer-format interface CSV.

    Columns: x_interface, D_med, D_lo, D_hi
    (D_lo and D_hi are synthetic ±20 % credible intervals.)
    """
    xi = np.linspace(0.5, 9.5, n_interfaces)
    D_med = _synthetic_D(xi)
    D_lo = D_med * 0.80
    D_hi = D_med * 1.20

    with open(path, "w") as fh:
        fh.write("x_interface,D_med,D_lo,D_hi\n")
        for i in range(n_interfaces):
            fh.write(
                f"{xi[i]:.18e},{D_med[i]:.18e},"
                f"{D_lo[i]:.18e},{D_hi[i]:.18e}\n"
            )
    print(f"  Wrote {path}  ({n_interfaces} interfaces)")


# ===================================================================
# 3.  Synthetic 2D FES — three-basin landscape
# ===================================================================
def _synthetic_2d_fes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Three-basin landscape on (CV₁, CV₂) ∈ [0, 8] × [0, 6].

    Basins centred at roughly:
        A ≈ (1.5, 1.5)   – deep  (global minimum)
        B ≈ (6.0, 1.5)   – intermediate
        C ≈ (4.0, 4.5)   – shallow

    Construction: quadratic bowl + three Gaussian wells + a ridge.
    """
    cx, cy = 4.0, 3.0  # centre of the envelope
    envelope = 0.08 * ((X - cx) ** 2 + (Y - cy) ** 2)

    wellA = 30.0 * np.exp(-(((X - 1.5) ** 2) / 2.0 + ((Y - 1.5) ** 2) / 1.8))
    wellB = 22.0 * np.exp(-(((X - 6.0) ** 2) / 1.6 + ((Y - 1.5) ** 2) / 1.4))
    wellC = 16.0 * np.exp(-(((X - 4.0) ** 2) / 1.4 + ((Y - 4.5) ** 2) / 1.2))

    # Diagonal ridge between A and C
    ridge = 6.0 * np.exp(-(((X - 2.8) + (Y - 3.0)) ** 2) / 0.6)

    F = envelope - wellA - wellB - wellC + ridge
    F -= np.nanmin(F)
    return F


def write_2d_fes(path: Path, nx: int = 201, ny: int = 151) -> None:
    x = np.linspace(0.0, 8.0, nx)
    y = np.linspace(0.0, 6.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")  # shape (nx, ny)
    F = _synthetic_2d_fes(X, Y)

    # Numerical gradient (for the derivative columns)
    dFdx = np.gradient(F, x, axis=0)
    dFdy = np.gradient(F, y, axis=1)

    with open(path, "w") as fh:
        fh.write("#! FIELDS cv1 cv2 file.free der_cv1 der_cv2\n")
        fh.write(f"#! SET min_cv1 {x[0]:.1f}\n")
        fh.write(f"#! SET max_cv1 {x[-1]:.1f}\n")
        fh.write(f"#! SET nbins_cv1  {nx}\n")
        fh.write("#! SET periodic_cv1 false\n")
        fh.write(f"#! SET min_cv2 {y[0]:.1f}\n")
        fh.write(f"#! SET max_cv2 {y[-1]:.1f}\n")
        fh.write(f"#! SET nbins_cv2  {ny}\n")
        fh.write("#! SET periodic_cv2 false\n")
        for i in range(nx):
            for j in range(ny):
                fh.write(
                    f"   {x[i]:.9f}   {y[j]:.9f}  "
                    f"{F[i, j]:.9f}   {dFdx[i, j]:.9f}   {dFdy[i, j]:.9f}\n"
                )
            if i < nx - 1:
                fh.write("\n")  # blank line between CV1 blocks
    print(f"  Wrote {path}  ({nx}×{ny} = {nx * ny} grid points)")


# ===================================================================
# Optional diagnostic plots
# ===================================================================
def _plot_diagnostics() -> None:
    import matplotlib.pyplot as plt

    s = np.linspace(0.0, 10.0, 501)
    F1 = _synthetic_1d_fes(s)
    D = _synthetic_D(s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax1.plot(s, F1, "k-", lw=1.5)
    ax1.set_ylabel("F  [kJ mol⁻¹]")
    ax1.set_title("Synthetic 1D FES — triple well")
    ax2.plot(s, D, "b-", lw=1.5)
    ax2.set_ylabel(r"$D(\xi)$  [CV² ps⁻¹]")
    ax2.set_xlabel(r"Reaction coordinate $\xi$")
    ax2.set_title(r"Synthetic $D(\xi)$")
    plt.tight_layout()
    plt.savefig(DATA / "diagnostic_1d.png", dpi=150)
    print(f"  Saved {DATA / 'diagnostic_1d.png'}")

    x = np.linspace(0.0, 8.0, 201)
    y = np.linspace(0.0, 6.0, 151)
    X, Y = np.meshgrid(x, y, indexing="ij")
    F2 = _synthetic_2d_fes(X, Y)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cs = ax.contourf(x, y, F2.T, levels=30, cmap="viridis_r")
    plt.colorbar(cs, ax=ax, label="F  [kJ mol⁻¹]")
    ax.set_xlabel(r"CV$_1$")
    ax.set_ylabel(r"CV$_2$")
    ax.set_title("Synthetic 2D FES — three basins")
    for (cx, cy), label in [((1.5, 1.5), "A"), ((6.0, 1.5), "B"), ((4.0, 4.5), "C")]:
        ax.plot(cx, cy, "wo", ms=8)
        ax.text(cx + 0.2, cy, label, color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DATA / "diagnostic_2d.png", dpi=150)
    print(f"  Saved {DATA / 'diagnostic_2d.png'}")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", action="store_true",
                        help="Save diagnostic PNG plots alongside the data files")
    args = parser.parse_args()

    print("Generating synthetic stochkin example data …")
    write_1d_fes(DATA / "synthetic_1d_fes.dat")
    write_diffusion_csv(DATA / "synthetic_diffusion_profile.csv")
    write_2d_fes(DATA / "synthetic_2d_fes.dat")

    if args.plot:
        _plot_diagnostics()

    print("\nDone.  Files are in:", DATA)


if __name__ == "__main__":
    main()
