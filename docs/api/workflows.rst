stochkin.workflows
==================

High-level, one-call wrappers for the most common stochastic-kinetics
pipelines.

The memory-corrected 1-D workflows require a user-supplied grid-level
``Sigma_t`` with shape ``(n_times, n_grid, n_grid)``.  Kernels are not
estimated or interpolated by ``stochkin``.

.. automodule:: stochkin.workflows
   :members:
   :undoc-members:
   :show-inheritance:
