#!/usr/bin/env python3
"""Example 01 – Analytic double-well potential: Langevin dynamics + MFPT
==========================================================================

This self-contained script demonstrates:
  - Defining an analytic 2D double-well potential
  - Detecting basins automatically from the potential landscape
  - Running parallel overdamped Langevin replicas
  - Computing bidirectional MFPT between the two basins

No external FES file is needed.  All quantities use reduced (dimensionless)
units: length in σ, time in τ, energy in ε = kT.

Run
---
    python 01_analytic_doublewell.py

Expected output (values are approximate)
-----------------------------------------
Basin 0 center ≈ (-1.0, -1.0)
Basin 1 center ≈ (+1.0, +1.0)
MFPT  0→1 ≈ O(10³)  τ
MFPT  1→0 ≈ O(10³)  τ
"""
import numpy as np
import matplotlib.pyplot as plt
import stochkin as sk

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
KT       = 0.5       # thermal energy (= 1/beta); barrier height / kT ≈ 5
GAMMA    = 1.0       # friction coefficient
DT       = 1e-3      # Langevin time step
MAX_TIME = 5e4       # max trajectory length per replica
N_TRIALS = 200       # number of parallel replicas

# ---------------------------------------------------------------------------
# Potential
# ---------------------------------------------------------------------------
def double_well(x):
    """2D double well U = (x0^2 - 1)^2 + x1^2."""
    x = np.asarray(x, dtype=float)
    U = (x[0]**2 - 1.0)**2 + x[1]**2
    F = np.array([
        -4.0 * x[0] * (x[0]**2 - 1.0),
        -2.0 * x[1],
    ])
    return U, F


# ---------------------------------------------------------------------------
# Basin detection
# ---------------------------------------------------------------------------
print("Detecting basins …")
basin_net = sk.build_basin_network_from_potential(
    double_well,
    xlim=(-2.0, 2.0),
    ylim=(-1.5, 1.5),
    nx=101,   # odd → x=±1.0 land exactly on grid nodes
    ny=101,   # odd → y=0.0 lands exactly on a grid node
)
print(f"  Found {basin_net.n_basins} basins")
for b in basin_net.basins:
    print(f"  Basin {b.id}: minimum ≈ {b.minimum}")

# ---------------------------------------------------------------------------
# Bidirectional MFPT (trajectory-based)
# ---------------------------------------------------------------------------
print("\nComputing MFPT via overdamped Langevin replicas …")
b0, b1 = basin_net.basins[0], basin_net.basins[1]

result = sk.compute_bidirectional_mfpt(
    double_well,
    lambda x: basin_net.which_basin(x) == b0.id,   # callable basin predicate
    lambda x: basin_net.which_basin(x) == b1.id,   # callable basin predicate
    kT=KT,
    gamma=GAMMA,
    n_trials=N_TRIALS,
    max_time=MAX_TIME,
    dt=DT,
    regime="overdamped",
    boundsA=b0.bounds,
    boundsB=b1.bounds,
    processes=1,          # lambdas aren't picklable → single-process
)
mfpt_01 = result["A_to_B"]["mean"]
mfpt_10 = result["B_to_A"]["mean"]
print(f"  MFPT  0→1 = {mfpt_01:.2e}  τ")
print(f"  MFPT  1→0 = {mfpt_10:.2e}  τ")

# ---------------------------------------------------------------------------
# Quick visualisation
# ---------------------------------------------------------------------------
X = np.linspace(-2.0, 2.0, 200)
Y = np.linspace(-1.5, 1.5, 200)
Xg, Yg = np.meshgrid(X, Y, indexing="ij")
U_grid = np.vectorize(lambda xi, yi: double_well([xi, yi])[0])(Xg, Yg)

fig, ax = plt.subplots(figsize=(5, 3.5))
cs = ax.contourf(X, Y, U_grid.T / KT, levels=np.linspace(0, 10, 20), cmap="viridis_r")
plt.colorbar(cs, ax=ax, label=r"$U / k_\mathrm{B}T$")
for b in basin_net.basins:
    mx, my = float(b.minimum[0]), float(b.minimum[1])
    ax.plot(mx, my, "o", color="white", ms=8)
    ax.text(mx, my, f" B{b.id}", color="white", va="center", fontsize=9)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Double-well potential")
plt.tight_layout()
plt.savefig("01_doublewell.png", dpi=150)
print("\nSaved 01_doublewell.png")
