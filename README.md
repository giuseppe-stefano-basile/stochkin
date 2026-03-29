# stochkin

**Stochastic kinetics toolkit** — Fokker–Planck / CTMC analysis on
1D and 2D free-energy surfaces from MD simulations.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

`stochkin` connects metadynamics free-energy surfaces (FES) to
continuous-time Markov chain (CTMC) kinetics via a Fokker–Planck
(Smoluchowski) operator built on the FES grid.  The key ingredients are:

| Step | Description |
|------|-------------|
| **FES input** | PLUMED-format 1D or 2D `.dat` files |
| **D(s) input** | Constant scalar **or** Hummer (2005) Bayesian position-dependent profile |
| **Basin detection** | Automatic local-minima labelling on a 1D or 2D grid |
| **FPE backward BVP** | Tridiagonal (1D) or FiPy (2D) solve for exit times & committors |
| **CTMC generator** | Rate matrix **K**, branching probabilities, mean first-passage times |
| **MFEP** | Grid-Dijkstra + NEB path refinement → 1D profile along arc-length |
| **Langevin dynamics** | BAOAB (underdamped) and overdamped BD replicas with multiprocessing |

---

## Installation

```bash
# Editable install from source (recommended while in active development)
git clone https://github.com/giuseppe-invernizzi/stochkin.git
cd stochkin
pip install -e .

# Optional: 2D FPE backend (FiPy)
pip install -e ".[fipy]"

# Dev tools
pip install -e ".[dev]"
```

**Dependencies**: `numpy`, `scipy`, `matplotlib`, `tqdm`, `pandas`
(all listed in `pyproject.toml`).

---

## Quick start

### 1 – 1D FES from PLUMED + constant D

```python
import stochkin as sk

result = sk.run_1d_ctmc_from_plumed(
    "fes.dat",
    D=0.04,            # CV² ps⁻¹
    T=300.0,
    crop=(4.5, 6.5),   # restrict CV range
    resample_n=500,
)

print("Rate matrix (ps⁻¹):\n", result["K_ps"])
print("Exit times (ns):",       result["exit_ps"] / 1000)
```

### 2 – 1D FES + Hummer D(s) profile

```python
result = sk.run_1d_ctmc_with_hummer_D(
    "fes.dat",
    d_csv="diffusion_profile.csv",  # columns: x_interface, D_med
    T=300.0,
    crop=(4.5, 6.5),
    resample_n=500,
)
```

### 3 – 2D FES → MFEP → 1D CTMC

```python
result = sk.run_mfep_ctmc(
    "fes_2d.dat",
    start=(1.0, 0.5),
    end=(5.0, 0.5),
    D_s=0.04,
    T=300.0,
)
mfep = result["mfep_path"]   # MFEPPath object with .s, .F, .x, .y
```

### 4 – Analytic potential + Langevin MFPT

```python
from stochkin import (
    double_well_2d,
    build_basin_network_from_potential,
    compute_bidirectional_mfpt,
)
import numpy as np

kT = 0.5
gamma = 1.0
basin_net = build_basin_network_from_potential(
    double_well_2d, xlim=(-2, 2), ylim=(-2, 2)
)
mfpt = compute_bidirectional_mfpt(
    double_well_2d, basin_net.basins[0], basin_net.basins[1],
    kT=kT, gamma=gamma, n_replicas=200, max_time=1e5, dt=1e-3,
)
```

---

## Module overview

| Module | Key symbols |
|--------|-------------|
| `workflows` | `run_1d_ctmc`, `run_1d_ctmc_from_plumed`, `run_1d_ctmc_with_hummer_D`, `run_mfep_ctmc`, `interface_to_centers`, `interpolate_D_to_grid` |
| `fes` | `load_plumed_fes_1d`, `load_plumed_fes_2d`, `FESPotential`, `FESPotential1D` |
| `potentials` | `muller_potential`, `double_well_2d`, `build_basin_network_from_fes_1d`, `BasinNetwork1D` |
| `fpe` | `compute_ctmc_generator_fpe_1d`, `compute_ctmc_generator_fpe_fipy`, `build_fp_generator_from_fes` |
| `mfep` | `compute_mfep_profile_1d`, `GridMFEP`, `NEBMFEP`, `MFEPPath` |
| `mfpt` | `compute_mfpt_network`, `compute_bidirectional_mfpt`, `compute_mfpt_network_fpe` |
| `integrators` | `baobab_2d`, `overdamped_bd` |
| `committor` | `committor_map_parallel`, `committor_map_fpe`, `committor_profile_1d` |
| `replicas` | `run_replicas`, `run_replicas_1d` |
| `uncertainty` | `bootstrap_ctmc_1d`, `bootstrap_ctmc_with_hummer_D`, `UncertaintyResult` |
| `plotting` | `plot_basin_network`, `plot_mfpt_matrix`, `plot_fp_solution_vs_boltzmann` |

---

## Result dictionary

All `run_*` workflow functions return a `dict` with:

| Key | Units | Description |
|-----|-------|-------------|
| `s` | CV | Uniform grid |
| `F` | kJ/mol | Free energy |
| `D_used` | CV²/time_unit | Diffusion used |
| `K` | 1/time_unit | Rate matrix |
| `K_ps` | 1/ps | Rate matrix in ps |
| `exit_mean` | time_unit | Mean exit time per basin |
| `exit_ps` | ps | Mean exit time in ps |
| `k_out` | 1/time_unit | Total exit rate per basin |
| `k_out_ps` | 1/ps | Total exit rate in ps |
| `p_branch` | – | Branching probability matrix |
| `labels_full` | – | Basin label per grid point (-1 = transition) |
| `basin_ids` | – | Sorted basin ids |

---

## Background

The kinetic framework is the overdamped Smoluchowski equation

$$\frac{\partial p}{\partial t} = \nabla \cdot \left[ D(s) \left( \nabla p + \beta \nabla F(s)\, p \right) \right]$$

Mean first-passage times and committor probabilities are obtained from
the corresponding backward equations (Robin / Dirichlet BVPs solved via
a tridiagonal sweep in 1D, or FiPy for 2D).  Position-dependent D(s)
profiles are supported via the Hummer (2005) Bayesian estimator.

**Reference:**  
G. Hummer, *Position-dependent diffusion coefficients and free energies
from Bayesian analysis of equilibrium and replica molecular dynamics
simulations*, New J. Phys. **7**, 34 (2005).

---

## Tests

```bash
pip install -e ".[test]"
pytest tests/
```

---

## License

MIT – see [LICENSE](LICENSE).
