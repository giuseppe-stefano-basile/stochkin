Experimental memory-kernel kinetics
===================================

User-supplied memory-kernel tools for GME propagation and memory-corrected
CTMC rate construction.

.. warning::

   This is a development feature.  Names, defaults, numerical methods, and
   result schemas may change between minor releases.  Import new code from
   ``stochkin.experimental.memory``.  Top-level aliases are retained for the
   current development cycle only.

.. note::

   ``stochkin`` does not estimate memory kernels from trajectories in this
   module.  Provide ``Sigma_t`` on the exact final grid with shape
   ``(n_times, n_grid, n_grid)`` and units ``time^-2`` together with
   ``memory_times`` in the same time unit as the Markovian generator ``K0``.
   Row-vector dynamics are the default, so rows of ``K0`` and each
   ``Sigma_t[k]`` should sum to zero.

The high-level rate workflow is
:func:`stochkin.experimental.memory.run_memory_corrected_ctmc_1d`: build ``K0`` from
``F(s), D(s)``, compute a moment-resummed effective grid generator, and
coarse-grain it into basin CTMC rates.  :func:`stochkin.experimental.memory.run_gme_1d`
is the lower-level propagation and transition-matrix validation API.

.. automodule:: stochkin.memory
   :members:
   :undoc-members:
   :show-inheritance:

Experimental namespace
----------------------

.. automodule:: stochkin.experimental.memory
   :members:
