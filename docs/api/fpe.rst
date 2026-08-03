stochkin.fpe
============

Fokker–Planck equation solvers and grid-based kinetic operators.

The 1-D utilities include both the standard Smoluchowski discretization
(:func:`stochkin.fpe.build_smolu_generator_1d`) and finite-state
coarse-graining from any grid generator with matching basin labels
(:func:`stochkin.fpe.compute_ctmc_generator_from_grid_generator_1d`).  The
latter is used by the memory-corrected CTMC workflow after a user-supplied
kernel has been converted into an effective grid generator.

.. automodule:: stochkin.fpe
   :members:
   :undoc-members:
   :show-inheritance:
