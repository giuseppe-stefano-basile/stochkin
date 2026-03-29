Getting Started
===============

Installation
------------

From the repository root:

.. code-block:: bash

   pip install -e .

This installs **stochkin** in editable mode with the core dependencies
(NumPy, Matplotlib).  For the full feature set install the optional
extras:

.. code-block:: bash

   pip install -e ".[full]"
   # or individually:
   pip install scipy fipy tqdm pandas

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
   * - Matplotlib
     - **yes**
     - plotting
   * - SciPy
     - optional
     - spline interpolation, sparse solvers
   * - FiPy
     - optional
     - 2-D Fokker–Planck PDE solves
   * - tqdm
     - optional
     - progress bars
   * - pandas
     - optional
     - CSV loading in workflows
