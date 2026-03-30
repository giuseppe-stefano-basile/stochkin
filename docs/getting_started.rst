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
