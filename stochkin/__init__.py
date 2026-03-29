"""
stochkin – stochastic kinetics toolkit
=======================================

Fokker–Planck / CTMC analysis on 1D and 2D free-energy surfaces,
Langevin dynamics, MFPT estimation, and minimum free-energy path tools.

Submodules
----------
workflows
    High-level one-call wrappers: :func:`run_1d_ctmc`,
    :func:`run_1d_ctmc_from_plumed`, :func:`run_1d_ctmc_with_hummer_D`,
    :func:`run_mfep_ctmc`.
potentials
    Analytic 2D model potentials (Müller–Brown, Mexican hat, etc.) and
    basin-detection utilities (Basin, BasinNetwork).
integrators
    BAOAB Langevin integrator and basic velocity/position updates.
fes
    FES loader from PLUMED-style tables and interpolated FES potentials.
fpe
    Fokker–Planck solvers and generator builders.
mfep
    Minimum free-energy path search (grid Dijkstra + NEB refinement).
replicas
    Parallel replica dynamics on analytic or FES-derived potentials.
committor
    Committor map calculation from trajectories (and optionally FPE).
mfpt
    MFPT and rate estimation from trajectories or FPE on the FES grid.
plotting
    Generic plotting helpers (FES, MFPT matrices, basin networks, etc.).
"""

__version__ = "0.1.0"

# --------------------------------------------------------------------
# Analytic potentials & basin partitioning
# --------------------------------------------------------------------
from .potentials import (
    # Analytic potentials
    make_potential_from_string,
    double_well_2d,
    simple_double_well_2d,
    mexican_hat_potential,
    central_well_barrier_ring_potential,
    muller_potential,
    # Grid sampling + basins
    sample_potential_grid,
    build_basin_network_from_potential,
    build_basin_network_from_fes_1d,
    build_basin_network_from_potential_1d,
    detect_basins_for_mfpt_1d,
    detect_basins_for_mfpt,
    # Data structures
    Basin,
    BasinNetwork,
    Basin1D,
    BasinNetwork1D,
    
)

# --------------------------------------------------------------------
# FES utilities (PLUMED tables, interpolated potentials)
# --------------------------------------------------------------------
from .fes import (
    SCIPY_AVAILABLE,
    load_plumed_fes_2d,
    load_plumed_fes_1d,
    make_fes_potential_from_grid,
    make_fes_potential_from_plumed,
    make_fes_potential_from_plumed_1d,
    plot_fes_colormap,
    FESPotential,
    FESPotential1D,
    # New: generic scalar field / diffusion loaders
    load_plumed_scalar_field_2d,
    load_diffusion_scalar_2d_from_plumed,
)

# --------------------------------------------------------------------
# MFEP utilities (2D -> 1D path projection)
# --------------------------------------------------------------------
from .mfep import (
    MFEPPath,
    GridMFEP,
    NEBMFEP,
    compute_mfep_profile_1d,
)

# --------------------------------------------------------------------
# Fokker–Planck solvers
# --------------------------------------------------------------------
from .fpe import (
    solve_fp_steady_state,
    # New: discrete FP generator on the 2D FES grid
    build_fp_generator_from_fes,
    compute_ctmc_generator_fpe_fipy,
    # 1D backward CTMC (NumPy; no FiPy)
    compute_ctmc_generator_fpe_1d,
)

# --------------------------------------------------------------------
# Langevin integrators
# --------------------------------------------------------------------
from .boundaries import apply_bounds, reflect_scalar, reflect_position_velocity

from .integrators import (
    velocity_update,
    position_update,
    random_velocity_update,
    baobab_2d,
)

# --------------------------------------------------------------------
# Parallel replica dynamics
# --------------------------------------------------------------------
from .replicas import (
    single_replica,
    run_replicas,
    single_replica_1d,
    run_replicas_1d,

)

# --------------------------------------------------------------------
# Committor analysis
# --------------------------------------------------------------------
from .committor import (
    run_short_trajectory,
    committor_point,
    committor_map_parallel,
    committor_profile_1d,
    basinA,
    basinB,
    # New: grid-based committor from FPE
    committor_map_fpe,
)

# --------------------------------------------------------------------
# MFPT and rate estimation
# --------------------------------------------------------------------
from .mfpt import (
    # Two-basin, trajectory-based MFPT
    single_passage_time,
    generate_basin_position,
    generate_basin_position_1d,
    compute_mfpt,
    compute_mfpt_1d,
    compute_bidirectional_mfpt,
    compute_bidirectional_mfpt_1d,
    plot_mfpt_statistics,
    estimate_transition_rates,
    basinA_mfpt,
    basinB_mfpt,
    # Multi-basin MFPT from trajectories
    compute_mfpt_network,
    estimate_rate_matrix,
    # New: multi-basin MFPT from FPE on the FES grid
    compute_mfpt_network_fpe,
)

# --------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------
from .plotting import (
    plot_results,
    plot_mfpt_matrix,
    plot_fp_solution_vs_boltzmann,
    plot_basin_network,
    plot_central_well_barrier_ring,
)

# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
__all__ = [
    # Potentials / basins
    "make_potential_from_string",
    "double_well_2d",
    "simple_double_well_2d",
    "mexican_hat_potential",
    "central_well_barrier_ring_potential",
    "muller_potential",
    "sample_potential_grid",
    "build_basin_network_from_potential",
    "detect_basins_for_mfpt",
    "Basin",
    "BasinNetwork",
    # FES utilities
    "SCIPY_AVAILABLE",
    "load_plumed_fes_2d",
    "load_plumed_fes_1d",
    "make_fes_potential_from_grid",
    "make_fes_potential_from_plumed",
    "make_fes_potential_from_plumed_1d",
    "plot_fes_colormap",
    "FESPotential",
    "FESPotential1D",
    "load_plumed_scalar_field_2d",
    "load_diffusion_scalar_2d_from_plumed",
    # MFEP
    "MFEPPath",
    "GridMFEP",
    "NEBMFEP",
    "compute_mfep_profile_1d",
    # FPE
    "solve_fp_steady_state",
    "build_fp_generator_from_fes",
    "compute_ctmc_generator_fpe_fipy",
    # Integrators
    "velocity_update",
    "position_update",
    "random_velocity_update",
    "baobab_2d",
    # Replicas
    "single_replica",
    "run_replicas",
    "single_replica_1d",
    "run_replicas_1d",
    # Committor
    "run_short_trajectory",
    "committor_point",
    "committor_map_parallel",
    "committor_profile_1d",
    "basinA",
    "basinB",
    "committor_map_fpe",
    # MFPT / rates
    "single_passage_time",
    "generate_basin_position",
    "generate_basin_position_1d",
    "compute_mfpt",
    "compute_mfpt_1d",
    "compute_bidirectional_mfpt",
    "compute_bidirectional_mfpt_1d",
    "plot_mfpt_statistics",
    "estimate_transition_rates",
    "basinA_mfpt",
    "basinB_mfpt",
    "compute_mfpt_network",
    "estimate_rate_matrix",
    "compute_mfpt_network_fpe",
    # Plotting
    "plot_results",
    "plot_mfpt_matrix",
    "plot_fp_solution_vs_boltzmann",
    "plot_basin_network",
    "plot_central_well_barrier_ring",
]

# --------------------------------------------------------------------
# High-level workflow wrappers
# --------------------------------------------------------------------
from .workflows import (
    run_1d_ctmc,
    run_1d_ctmc_from_plumed,
    run_1d_ctmc_with_hummer_D,
    run_mfep_ctmc,
    run_multi_mfep_ctmc,
    interface_to_centers,
    interpolate_D_to_grid,
)

# Append to __all__
__all__ += [
    "run_1d_ctmc",
    "run_1d_ctmc_from_plumed",
    "run_1d_ctmc_with_hummer_D",
    "run_mfep_ctmc",
    "run_multi_mfep_ctmc",
    "interface_to_centers",
    "interpolate_D_to_grid",
]

# --------------------------------------------------------------------
# Publication style
# --------------------------------------------------------------------
from .style import (
    set_publication_style,
    publication_style,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    CBAR_LABEL_SIZE,
    CBAR_TICK_SIZE,
    TITLE_SIZE,
)
from .plotting import plot_2d_fes, draw_barrier_arrows

__all__ += [
    # Style
    "set_publication_style",
    "publication_style",
    "LABEL_SIZE",
    "TICK_SIZE",
    "LEGEND_SIZE",
    "CBAR_LABEL_SIZE",
    "CBAR_TICK_SIZE",
    "TITLE_SIZE",
    # Extra plotting
    "plot_2d_fes",
    "draw_barrier_arrows",
]

# --------------------------------------------------------------------
# Uncertainty propagation
# --------------------------------------------------------------------
from .uncertainty import (
    UncertaintyResult,
    bootstrap_ctmc_1d,
    bootstrap_ctmc_with_hummer_D,
)

__all__ += [
    "UncertaintyResult",
    "bootstrap_ctmc_1d",
    "bootstrap_ctmc_with_hummer_D",
]
