"""Experimental memory-kernel kinetics API.

This is the supported discovery point for the development version of the
memory-kernel functionality.  The implementation remains split across
``stochkin.memory``, ``stochkin.fpe``, and ``stochkin.workflows`` so it can be
promoted without moving numerical code later.

Notes
-----
This API is experimental.  Names, defaults, numerical methods, and result
schemas may change between minor releases.  The current implementation only
accepts user-supplied kernels; it does not infer a kernel from trajectories.
"""

from ..fpe import (
    build_smolu_generator_1d,
    compute_ctmc_generator_from_grid_generator_1d,
)
from ..memory import (
    EffectiveGeneratorResult,
    GMEPropagationResult,
    MemoryKernelInput,
    MemoryValidationResult,
    chapman_kolmogorov_error,
    effective_markov_generator_from_memory,
    enforce_generator_conservation,
    memory_corrected_generator,
    memory_moments,
    propagate_gme,
    propagate_gme_transition_matrix,
    validate_memory_kernel,
    validate_memory_model,
)
from ..workflows import (
    run_gme_1d,
    run_memory_corrected_ctmc_1d,
    run_memory_corrected_ctmc_from_plumed,
)

EXPERIMENTAL_API = True
EXPERIMENTAL_API_VERSION = "0.1"

__all__ = [
    "EXPERIMENTAL_API",
    "EXPERIMENTAL_API_VERSION",
    "MemoryKernelInput",
    "MemoryValidationResult",
    "GMEPropagationResult",
    "EffectiveGeneratorResult",
    "build_smolu_generator_1d",
    "compute_ctmc_generator_from_grid_generator_1d",
    "validate_memory_kernel",
    "enforce_generator_conservation",
    "propagate_gme",
    "propagate_gme_transition_matrix",
    "memory_moments",
    "effective_markov_generator_from_memory",
    "memory_corrected_generator",
    "chapman_kolmogorov_error",
    "validate_memory_model",
    "run_gme_1d",
    "run_memory_corrected_ctmc_1d",
    "run_memory_corrected_ctmc_from_plumed",
]
