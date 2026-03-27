import numpy as np

from .boundaries import apply_bounds as _apply_bounds

# =============================================================================
# Underdamped BAOAB (kept for backwards compatibility)
# =============================================================================

def velocity_update(v, F, dt, m=1.0):
    """Half-kick (B) step."""
    return v + (F / m) * (dt / 2.0)

def position_update(x, v, dt):
    """Half-drift (A) step."""
    return x + v * (dt / 2.0)

def random_velocity_update(v, gamma, kT, dt, m=1.0):
    """Ornstein–Uhlenbeck (O) step."""
    R = np.random.randn(len(v))
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1.0 - c1**2) * kT / m)
    return c1 * v + c2 * R

def baobab_step(potential, x, v, dt, gamma, kT, m=1.0):
    """
    One BAOAB step. Shared by mfpt/committor to avoid inconsistent drifts.
    potential(x)->(U,F) with F=-∇U.
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
    """
    BAOAB Langevin integrator (works in any dimension despite the name).
    Returns times, positions, velocities, energies.
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
    """
    Picklable adapter: D(x) = kT / gamma(x).
    gamma may be scalar or callable (must be picklable for multiprocessing).
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
    """
    Evaluate scalar D(x) and ∇D(x).
    diffusion can be:
      - scalar
      - callable D(x)
      - object with __call__(x)->D and grad(x)->∇D (preferred)
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
    """
    Euler–Maruyama step consistent with divergence-form FPE:
      ∂_t p = ∇·[ D(x) ( ∇p + β p ∇U ) ]
    => drift b(x) = β D(x) F(x) + ∇D(x), with F=-∇U.
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
    """
    Overdamped trajectory generator.
    Provide diffusion (preferred) OR gamma (then D=kT/gamma).
    Returns (times, positions, velocities, energies) for API compatibility.
    velocities are zeros; energies are U(x) only.
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
