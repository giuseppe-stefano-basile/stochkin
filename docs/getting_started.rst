Getting Started
===============

Installation
------------

From the repository root:

.. code-block:: bash

   pip install -e .

This installs **stochkin** in editable mode with the core dependencies
(NumPy, SciPy, Matplotlib, tqdm, pandas). Optional extras are available
for notebook usage, development tooling, and the FiPy-based 2-D Fokker–
Planck backend:

.. code-block:: bash

   pip install -e ".[notebooks]"   # Jupyter + ipykernel
   pip install -e ".[dev]"         # pytest, black, ruff, pre-commit
   pip install -e ".[fipy]"        # optional 2-D FPE backend

If you only need notebooks without editable install extras, a minimal
alternative is:

.. code-block:: bash

   pip install jupyter ipykernel

Bundled examples and notebooks
------------------------------

The repository includes both script and notebook versions of the main
worked examples:

- ``examples/generate_synthetic_data.py`` plus
  ``notebooks/00_generate_synthetic_data.ipynb``
- ``examples/01_analytic_doublewell.py`` plus
  ``notebooks/01_analytic_doublewell.ipynb``
- ``examples/02_1d_plumed_fes_ctmc.py`` plus
  ``notebooks/02_1d_plumed_fes_ctmc.ipynb``
- ``examples/03_1d_hummer_D_ctmc.py`` plus
  ``notebooks/03_1d_hummer_D_ctmc.ipynb``
- ``examples/04_mfep_ctmc.py`` plus
  ``notebooks/04_mfep_ctmc.ipynb``
- ``examples/05_pairwise_mfep_paths.py`` plus
  ``notebooks/05_pairwise_mfep_paths.ipynb``
- ``examples/06_uncertainty.py`` plus
  ``notebooks/06_uncertainty.ipynb``
- ``examples/08_user_memory_gme_1d.py`` demonstrates GME propagation
  with a user-supplied memory kernel
- ``examples/09_memory_corrected_ctmc_rates.py`` demonstrates memory-corrected
  basin CTMC rates from a user-supplied grid-level kernel

The notebooks are generated from the template builder:

.. code-block:: bash

   python tools/build_example_notebooks.py

Most examples use the bundled synthetic datasets in ``examples/data/``.
If you want to regenerate those files first, run:

.. code-block:: bash

   python examples/generate_synthetic_data.py --plot

Minimal example
---------------

Compute CTMC rates along a 1-D free-energy profile loaded from a
PLUMED ``sum_hills`` output file:

.. code-block:: python

   from stochkin.workflows import run_1d_ctmc_from_plumed

   result = run_1d_ctmc_from_plumed(
       fes_path="fes_1d.dat",
       D=0.05,          # diffusion coefficient [CV²/ps]
       T=300.0,         # temperature [K]
       time_unit="ps",
   )

   print("Rate matrix [1/ps]:")
   print(result["K_ps"])
   print("Mean exit times [ps]:", result["exit_ps"])

Key concepts
------------

Potential callable
^^^^^^^^^^^^^^^^^^

Every potential in stochkin is a *callable* with signature::

    U, F = potential(x)

where ``x`` is a position vector (``ndarray``), ``U`` is the scalar
energy, and ``F = −∇U`` is the force vector.  All analytic potentials
in :mod:`stochkin.potentials` and the FES interpolators in
:mod:`stochkin.fes` follow this convention.

Basin network
^^^^^^^^^^^^^

A :class:`~stochkin.potentials.BasinNetwork` (or its 1-D counterpart
:class:`~stochkin.potentials.BasinNetwork1D`) groups the grid into
metastable basins separated by barriers.  It is the input to MFPT
network estimation, CTMC construction, and committor analysis.

CTMC generator
^^^^^^^^^^^^^^

The continuous-time Markov chain (CTMC) generator :math:`K` is an
:math:`n \times n` matrix whose off-diagonal element :math:`K_{ij}`
is the rate of transition from basin *i* to basin *j*, and the diagonal
satisfies :math:`K_{ii} = -\sum_{j \neq i} K_{ij}` so that rows sum
to zero.

Memory-corrected CTMC rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   Memory-kernel kinetics is an experimental development feature.  Import it
   from ``stochkin.experimental.memory`` and expect API and numerical-method
   changes between minor releases.

The memory workflow extends the standard 1-D CTMC construction instead of
replacing it.  First ``F(s)`` and ``D(s)`` define the Markovian
Smoluchowski grid generator ``K0``.  Then a user-supplied many-body-inspired
kernel :math:`\Sigma(t)` corrects that grid generator through memory moments.
Finally, the corrected grid generator is coarse-grained into basin-to-basin
CTMC rates with the same backward-equation logic used by the Markovian
workflow.

``stochkin`` does not estimate :math:`\Sigma(t)` from trajectories.  Supply
``Sigma_t`` on the exact final CV grid with shape
``(n_times, n_grid, n_grid)`` and units ``time^-2``.  The ``memory_times``
array must use the same time unit as ``D`` and the returned rates.  No
matrix-valued interpolation is performed by the PLUMED wrapper; if you crop
or resample the FES, provide a kernel for that final grid.

By default, stochkin uses row-vector convention: ``p(t) = p(0) T(t)``.
Therefore ``K0``, the effective generator, and each ``Sigma_t[k]`` should
have rows summing to zero.  Use ``convention="column"`` only when all
matrices are supplied in column-vector orientation.

.. code-block:: python

   import numpy as np
   from stochkin.experimental import memory as memory_kinetics

   s = np.linspace(-1.5, 1.5, 41)
   F = 0.35 * (s**2 - 1.0) ** 2
   F -= F.min()

   D = 0.002
   K0 = memory_kinetics.build_smolu_generator_1d(s, F, D=D, beta=1.0)
   memory_times = np.linspace(0.0, 2.0, 201)
   Sigma_t = np.asarray([0.12 * np.exp(-t / 0.35) * K0 for t in memory_times])

   result = memory_kinetics.run_memory_corrected_ctmc_1d(
       s, F, D,
       Sigma_t=Sigma_t,
       memory_times=memory_times,
       memory_order=1,
       beta=1.0,
   )

   print(result["K"])
   print(result["memory_diagnostics"])

The lower-level
:func:`stochkin.experimental.memory.run_gme_1d` workflow remains useful for
propagating probability vectors or transition matrices when you want to
validate or visualize non-Markovian dynamics directly.


Uncertainty propagation
^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`stochkin.uncertainty` module propagates credible intervals on
*F(s)* and *D(s)* through the CTMC pipeline via Monte Carlo bootstrap.
Each replicate perturbs the inputs (Gaussian for F, log-normal for D),
re-runs the full BVP solver, and the resulting rates / exit times are
collected into confidence intervals.

.. code-block:: python

   import stochkin as sk

   res = sk.bootstrap_ctmc_1d(
       s, F, D,
       D_lo=D_lo_grid, D_hi=D_hi_grid,
       n_bootstrap=200,
       seed=42,
       T=300.0,
       time_unit="ps",
   )
   print(res.summary("ps"))
   # Access: res.K_ps_ci_lo, res.K_ps_ci_hi, res.exit_mean_ci_lo, ...

See :doc:`api/uncertainty` for the full API and ``examples/06_uncertainty.py``
for a complete worked example.

Dependencies
------------

.. list-table::
   :header-rows: 1

   * - Package
     - Required?
     - Used for
   * - NumPy
     - **yes**
     - arrays, linear algebra
   * - SciPy
     - **yes**
     - interpolation, sparse/numerical utilities
   * - Matplotlib
     - **yes**
     - plotting
   * - tqdm
     - **yes**
     - progress bars
   * - pandas
     - **yes**
     - CSV loading in workflows
   * - FiPy
     - optional
     - 2-D Fokker–Planck PDE solves
   * - jupyter, ipykernel
     - optional
     - running the bundled example notebooks
