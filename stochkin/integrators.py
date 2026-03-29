r"""stochkin.integrators
=====================

Langevin dynamics integrators and overdamped Brownian dynamics steppers.

This module provides the core time-stepping routines used throughout
stochkin for trajectory-based analyses (MFPT, committors, replicas).

Underdamped Langevin
--------------------
:func:`baobab_step` implements a single BAOAB splitting step
(Leimkuhler & Matthews, 2013).  BAOAB is a symmetric Strang splitting
of the Langevin equation into:

* **B** – half velocity kick (``velocity_update``)
* **A** – half position drift (``position_update``)
* **O** – Ornstein–Uhlenbeck thermostat (``random_velocity_update``)
* **A** – half position drift
* **B** – half velocity kick

:func:`baobab_2d` wraps the single step into a full trajectory loop
with save-frequency control.

Overdamped Brownian dynamics
----------------------------
:func:`overdamped_step_euler` performs a single Euler–Maruyama step
consistent with the divergence-form Fokker–Planck equation:

.. math::
    \partial_t p = \nabla \cdot \bigl[ D(x)(\nabla p + \beta\, p\, \nabla U) \bigr]

which yields the drift
:math:`b(x) = \beta D(x) F(x) + \nabla D(x)` with
:math:`F = -\nabla U`.

:func:`overdamped_bd` wraps the step into a trajectory loop with
burn-in, save-frequency, and optional bounding-box enforcement.

Diffusion helpers
-----------------
:class:`GammaToDiffusion` converts a (possibly position-dependent)
friction coefficient γ into a diffusion coefficient D = kT/γ.
:func:`eval_diffusion_and_grad` dispatches scalar / callable / object
forms of D(x) into a uniform (D, ∇D) interface.

References
----------
B. Leimkuhler and C. Matthews, *Appl. Math. Res. Express* **2013**, 34 (2013).
"""

import numpy as np

from .boundaries import apply_bounds as _apply_bounds

# =============================================================================
# Underdamped BAOAB (kept for backwards compatibility)
# =============================================================================

def velocity_update(v, F, dt, m=1.0):
    """Half-kick (**B** step) of the BAOAB splitting.

    Parameters
    ----------
    v : ndarray
        Current velocity.
    F : ndarray
        Force vector *F = −∇U*.
    dt : float
        Full time step (half-step is applied internally).
    m : float
        Particle mass (default 1).

    Returns
    -------
    v_new : ndarray
        Updated velocity after a half-kick.
    """
    return v + (F / m) * (dt / 2.0)

def position_update(x, v, dt):
    """Half-drift (**A** step) of the BAOAB splitting.

    Parameters
    ----------
    x : ndarray
        Current position.
    v : ndarray
        Current velocity.
    dt : float
        Full time step (half-step is applied internally).

    Returns
    -------
    x_new : ndarray
        Updated position after a half-drift.
    """
    return x + v * (dt / 2.0)

def random_velocity_update(v, gamma, kT, dt, m=1.0):
    """Ornstein–Uhlenbeck thermostat (**O** step) of the BAOAB splitting.

    Applies an exact integration of the Ornstein–Uhlenbeck process
    over one full time step *dt*, generating the correct Gaussian noise
    for the given friction γ and temperature kT.

    Parameters
    ----------
    v : ndarray
        Current velocity.
    gamma : float
        Friction coefficient.
    kT : float
        Thermal energy :math:`k_B T`.
    dt : float
        Time step.
    m : float
        Particle mass (default 1).

    Returns
    -------
    v_new : ndarray
        Thermostatted velocity.
    """
    R = np.random.randn(len(v))
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1.0 - c1**2) * kT / m)
    return c1 * v + c2 * R

def baobab_step(potential, x, v, dt, gamma, kT, m=1.0):
    """Perform one full BAOAB splitting step.

    This is the core time-stepper shared by MFPT, committor, and replica
    modules to ensure consistent dynamics across the library.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)`` with ``F = −∇U``.
    x : ndarray
        Current position.
    v : ndarray
        Current velocity.
    dt : float
        Integration time step.
    gamma : float
        Friction coefficient.
    kT : float
        Thermal energy :math:`k_B T`.
    m : float
        Particle mass (default 1).

    Returns
    -------
    x_new : ndarray
        Updated position.
    v_new : ndarray
        Updated velocity.
    U : float
        Potential energy at the new position.
    """
    U, F = potential(x)
    v = velocity_update(v, F, dt, m)
    x = position_update(x, v, dt)
    v = random_velocity_update(v, gamma, kT, dt, m)
    x = position_update(x, v, dt)
    U, F = potential(x)
    v = velocity_update(v, F, dt, m)
    return x, v, U

def baobab_2d(
    potential,
    max_time,
    dt,
    gamma,
    kT,
    initial_position,
    initial_velocity,
    save_frequency=10,
    m=1.0,
):
    """Run a full BAOAB Langevin trajectory.

    Despite the “2d” suffix (historical), this integrator works in any
    number of dimensions.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)`` with ``F = −∇U``.
    max_time : float
        Total simulation time.
    dt : float
        Integration time step.
    gamma : float
        Friction coefficient.
    kT : float
        Thermal energy.
    initial_position : array_like
        Starting position.
    initial_velocity : array_like
        Starting velocity.
    save_frequency : int
        Save a snapshot every *save_frequency* steps (default 10).
    m : float
        Particle mass (default 1).

    Returns
    -------
    times : ndarray
        Time stamps of saved snapshots.
    positions : ndarray, shape (n_saved, d)
        Saved positions.
    velocities : ndarray, shape (n_saved, d)
        Saved velocities.
    energies : ndarray
        Total energy (kinetic + potential) at each saved snapshot.
    """
    x = np.array(initial_position, dtype=float).ravel()
    v = np.array(initial_velocity, dtype=float).ravel()
    t = 0.0
    step_number = 0
    positions, velocities, energies, times = [], [], [], []

    while t < max_time:
        x, v, U = baobab_step(potential, x, v, dt, gamma, kT, m)

        if step_number % save_frequency == 0:
            E_total = 0.5 * m * np.dot(v, v) + U
            positions.append(x.copy())
            velocities.append(v.copy())
            energies.append(E_total)
            times.append(t)

        t += dt
        step_number += 1

    return np.array(times), np.array(positions), np.array(velocities), np.array(energies)

# =============================================================================
# Overdamped (Smoluchowski/Brownian) with possibly position-dependent D(x)
# =============================================================================

class GammaToDiffusion:
    r"""Adapter converting a friction coefficient γ to a diffusion coefficient.

    Computes :math:`D(x) = k_BT / \gamma(x)` where γ may be a scalar or
    a callable ``γ(x)``.  The object is picklable, making it safe for use
    with ``multiprocessing.Pool``.

    Parameters
    ----------
    gamma : float or callable
        Friction coefficient.  If callable, must accept a position vector
        and return a positive scalar.
    kT : float
        Thermal energy :math:`k_B T`.

    Examples
    --------
    >>> D = GammaToDiffusion(gamma=10.0, kT=0.05)
    >>> D(np.array([0.0, 0.0]))  # returns 0.005
    """
    def __init__(self, gamma, kT):
        self.gamma = gamma
        self.kT = float(kT)

    def __call__(self, x):
        g = float(self.gamma(x)) if callable(self.gamma) else float(self.gamma)
        if g <= 0.0:
            raise ValueError("gamma must be > 0.")
        return self.kT / g

def _as_1d_array(x):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Empty position.")
    return arr


def apply_bounds(x, bounds, mode="reflect"):
    """Enforce rectangular bounds on a position vector.

    This is a thin wrapper around :func:`Stochastic_Estimation.boundaries.apply_bounds`
    kept for backwards compatibility with older scripts that imported it from
    :mod:`integrators`.

    Parameters
    ----------
    x : array_like, shape (d,)
    bounds : sequence of (lo, hi), length d
    mode : {'reflect', 'clip'}
    """
    return _apply_bounds(x, bounds, mode=mode)

def finite_difference_grad_scalar(fun, x, eps=1e-6):
    """Central FD gradient of scalar fun(x) with x in R^d."""
    x = _as_1d_array(x)
    grad = np.zeros_like(x)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        grad[i] = (float(fun(x + dx)) - float(fun(x - dx))) / (2.0 * eps)
    return grad

def eval_diffusion_and_grad(diffusion, x, eps=1e-6):
    """Evaluate scalar diffusion D(x) and its gradient ∇D(x).

    Dispatches on the type of *diffusion*:

    * **scalar** – returns ``(D, zeros)``.
    * **callable with** ``.grad()`` method – uses analytic gradient.
    * **plain callable** – uses central finite differences (step *eps*).

    Parameters
    ----------
    diffusion : scalar, callable, or object
        The diffusion coefficient.
    x : ndarray
        Position vector.
    eps : float
        Step size for numerical gradient (default 1e-6).

    Returns
    -------
    D : float
        Diffusion value at *x*.
    grad_D : ndarray
        Gradient ∇D at *x*, same shape as *x*.

    Raises
    ------
    ValueError
        If *D* is negative.
    TypeError
        If *diffusion* is not a recognised type.
    """
    x = _as_1d_array(x)

    if diffusion is None:
        raise ValueError("diffusion must be provided for overdamped dynamics.")

    if np.isscalar(diffusion):
        D = float(diffusion)
        if D < 0.0:
            raise ValueError("Diffusion must be >= 0.")
        return D, np.zeros_like(x)

    if hasattr(diffusion, "grad") and callable(getattr(diffusion, "grad")):
        D = float(diffusion(x))
        if D < 0.0:
            raise ValueError("Diffusion must be >= 0.")
        g = np.asarray(diffusion.grad(x), dtype=float).ravel()
        if g.shape != x.shape:
            raise ValueError("diffusion.grad(x) must have same shape as x.")
        return D, g

    if callable(diffusion):
        D = float(diffusion(x))
        if D < 0.0:
            raise ValueError("Diffusion must be >= 0.")
        g = finite_difference_grad_scalar(diffusion, x, eps=eps)
        return D, g

    raise TypeError("diffusion must be a scalar, callable, or object with .grad().")

def overdamped_step_euler(potential, x, dt, beta, diffusion, eps=1e-6):
    r"""Single Euler–Maruyama step for overdamped Brownian dynamics.

    The step is consistent with the divergence-form Fokker–Planck equation:

    .. math::
        \partial_t p = \nabla \cdot \bigl[ D(x)(\nabla p + \beta\, p\, \nabla U) \bigr]

    which gives the drift :math:`b(x) = \beta D(x) F(x) + \nabla D(x)` and
    noise amplitude :math:`\sqrt{2 D(x)\,\mathrm{d}t}`.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)`` with ``F = −∇U``.
    x : ndarray
        Current position.
    dt : float
        Time step.
    beta : float
        Inverse thermal energy :math:`1/(k_BT)`.
    diffusion : scalar, callable, or object
        Diffusion coefficient *D(x)*.  See :func:`eval_diffusion_and_grad`.
    eps : float
        Finite-difference step for ∇D when *diffusion* has no ``.grad()``.

    Returns
    -------
    x_new : ndarray
        Updated position.
    """
    x = _as_1d_array(x)
    _, F = potential(x)
    F = _as_1d_array(F)

    D, gradD = eval_diffusion_and_grad(diffusion, x, eps=eps)
    drift = beta * D * F + gradD
    noise = np.sqrt(2.0 * D * dt) * np.random.randn(x.size)

    return x + drift * dt + noise

def overdamped_bd(
    potential,
    max_time,
    dt,
    kT,
    initial_position,
    diffusion=None,
    gamma=None,
    save_frequency=10,
    burn_in_steps=0,
    bounds=None,
    boundary="reflect",
    seed=None,
    eps=1e-6,
):
    """Run a full overdamped (Smoluchowski) Brownian-dynamics trajectory.

    Provide either *diffusion* (preferred) or *gamma* (then D = kT/γ).
    Trajectories can optionally be confined inside a bounding box via
    mirror-reflection or clipping.

    Parameters
    ----------
    potential : callable
        ``potential(x) -> (U, F)`` with ``F = −∇U``.
    max_time : float
        Total simulation time.
    dt : float
        Integration time step.
    kT : float
        Thermal energy :math:`k_B T`.
    initial_position : array_like
        Starting position.
    diffusion : scalar, callable, or object, optional
        Diffusion coefficient *D(x)*.  Mutually exclusive with *gamma*.
    gamma : float or callable, optional
        Friction coefficient.  Converted to *D = kT/γ* internally.
    save_frequency : int
        Save a snapshot every *save_frequency* steps.
    burn_in_steps : int
        Number of initial steps to discard before recording.
    bounds : sequence of (lo, hi), optional
        Rectangular bounding box per coordinate.
    boundary : {'reflect', 'clip'}
        How to enforce *bounds* (default ``'reflect'``).
    seed : int, optional
        Random-number seed for reproducibility.
    eps : float
        Finite-difference step for ∇D.

    Returns
    -------
    times : ndarray
        Time stamps of saved snapshots.
    positions : ndarray, shape (n_saved, d)
        Saved positions.
    velocities : ndarray
        Zeros (API-compatible with ``baobab_2d``).
    energies : ndarray
        Potential energy *U(x)* at each snapshot (no kinetic term).
    """
    beta = 1.0 / float(kT)

    if seed is not None:
        np.random.seed(int(seed) % (2**32 - 1))

    if diffusion is None:
        if gamma is None:
            raise ValueError("Provide either diffusion or gamma for overdamped_bd.")
        diffusion = GammaToDiffusion(gamma=gamma, kT=kT)

    x = np.array(initial_position, dtype=float).ravel()
    x = apply_bounds(x, bounds, mode=boundary)
    t = 0.0
    step_number = 0

    positions, energies, times = [], [], []

    while t < max_time:
        x = overdamped_step_euler(potential, x, dt, beta, diffusion, eps=eps)
        x = apply_bounds(x, bounds, mode=boundary)

        if step_number >= int(burn_in_steps) and (step_number - int(burn_in_steps)) % int(save_frequency) == 0:
            U, _ = potential(x)
            positions.append(x.copy())
            energies.append(float(U))
            times.append(t)

        t += dt
        step_number += 1

    positions = np.array(positions, dtype=float)
    velocities = np.zeros_like(positions)
    return np.array(times), positions, velocities, np.array(energies, dtype=float)
