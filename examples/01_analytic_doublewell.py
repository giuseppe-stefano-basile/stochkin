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
N_REP    = 200       # number of parallel replicas

RNG = np.random.default_rng(42)

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
    ns=100,
)
print(f"  Found {basin_net.n_basins} basins")
for b in basin_net.basins:
    print(f"  Basin {b.id}: center ≈ {b.center}")

# ---------------------------------------------------------------------------
# Bidirectional MFPT (trajectory-based)
# ---------------------------------------------------------------------------
print("\nComputing MFPT via Langevin replicas …")
result = sk.compute_bidirectional_mfpt(
    double_well,
    basin_net.basins[0],
    basin_net.basins[1],
    kT=KT,
    gamma=GAMMA,
    n_replicas=N_REP,
    max_time=MAX_TIME,
    dt=DT,
    regime="overdamped",
)
mfpt_01 = result.get("mfpt_AB", float("nan"))
mfpt_10 = result.get("mfpt_BA", float("nan"))
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
    ax.plot(*b.center, "o", color="white", ms=8)
    ax.text(*b.center, f" B{b.id}", color="white", va="center", fontsize=9)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Double-well potential")
plt.tight_layout()
plt.savefig("01_doublewell.png", dpi=150)
print("\nSaved 01_doublewell.png")
