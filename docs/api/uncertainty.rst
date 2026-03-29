stochkin.uncertainty
====================

Monte Carlo uncertainty propagation for CTMC kinetics.

This module provides bootstrap-style uncertainty estimation: given
credible intervals (or standard deviations) on the free energy *F(s)*
and the diffusion coefficient *D(s)*, it generates perturbed replicates,
runs the full 1-D CTMC pipeline on each, and collects statistics
(mean, std, confidence intervals) on rates, exit times, and branching
probabilities.

Key functions
-------------

.. autosummary::
   :toctree: _autosummary

   stochkin.uncertainty.bootstrap_ctmc_1d
   stochkin.uncertainty.bootstrap_ctmc_with_hummer_D
   stochkin.uncertainty.UncertaintyResult

Detailed API
------------

.. automodule:: stochkin.uncertainty
   :members:
   :undoc-members:
   :show-inheritance:
