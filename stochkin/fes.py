"""stochkin.fes
=============

Free-energy surface (FES) loading, interpolation, and plotting.

This module provides tools for working with free-energy surfaces
obtained from enhanced-sampling simulations (e.g. PLUMED metadynamics
or umbrella sampling).

**Loading**

- :func:`load_plumed_fes_2d` / :func:`load_plumed_fes_1d` – read
  PLUMED-style ``.dat`` files into NumPy arrays.
- :func:`load_plumed_scalar_field_2d` /
  :func:`load_diffusion_scalar_2d_from_plumed` – generic scalar-field
  loaders (e.g. position-dependent diffusion).

**Interpolating potential objects**

- :class:`FESPotential` – picklable 2D interpolating potential
  ``potential(x) -> (U, F)`` from a regular grid.  Supports spline
  (SciPy ``RectBivariateSpline``) and bilinear (pure NumPy) modes.
- :class:`FESPotential1D` – 1D analogue using ``numpy.interp`` +
  finite-difference forces.
- :func:`make_fes_potential_from_grid` /
  :func:`make_fes_potential_from_plumed` – convenience constructors.

**Plotting**

- :func:`plot_fes_colormap` – filled-contour colormap of a 2D FES.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Try to import scipy for FES functionality
try:
    from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, griddata
    from scipy.ndimage import label, center_of_mass
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: SciPy not available. FES functionality will be limited.")
    SCIPY_AVAILABLE = False


def load_plumed_fes_2d(
    filename,
    x_col=0,
    y_col=1,
    fes_col=2,
    subtract_min=True,
    verbose=True,
):
    """Load a 2D free-energy surface from a PLUMED-style ``.dat`` file.

    The file is expected to have at least three numeric columns
    (``x``, ``y``, ``F(x,y)``) with ``#``-comment lines.

    Parameters
    ----------
    filename : str
        Path to the text file.
    x_col, y_col, fes_col : int
        Zero-based column indices.
    subtract_min : bool
        If ``True``, shift so that :math:`\\min F = 0`.
    verbose : bool
        Print grid dimensions.

    Returns
    -------
    x_grid, y_grid : ndarray
        Sorted 1D coordinate arrays.
    fes_grid : ndarray, shape (nx, ny)
        Free-energy values on the regular grid (may contain NaN for
        incomplete grids).
    """
    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        data = data[None, :]

    ncols = data.shape[1]
    if not (0 <= x_col < ncols and 0 <= y_col < ncols and 0 <= fes_col < ncols):
        raise ValueError(
            f"Column indices out of range for file with {ncols} columns: "
            f"x_col={x_col}, y_col={y_col}, fes_col={fes_col}"
        )

    x = data[:, x_col]
    y = data[:, y_col]
    fes = data[:, fes_col]

    if subtract_min:
        fes = fes - np.nanmin(fes)

    # Build unique coordinate grids (sorted)
    x_grid = np.unique(x)
    y_grid = np.unique(y)
    nx = x_grid.size
    ny = y_grid.size

    fes_grid = np.full((nx, ny), np.nan)

    # Map each (x, y) to indices on the grid
    ix = np.searchsorted(x_grid, x)
    iy = np.searchsorted(y_grid, y)

    if (
        np.any(ix < 0)
        or np.any(ix >= nx)
        or np.any(iy < 0)
        or np.any(iy >= ny)
    ):
        raise ValueError(
            "Some FES data points fall outside the inferred (x,y) grid."
        )

    for f_val, i, j in zip(fes, ix, iy):
        fes_grid[i, j] = f_val

    missing = np.isnan(fes_grid).sum()
    if verbose:
        print(f"Loaded FES from '{filename}':")
        print(f"  nx = {nx}, ny = {ny}, total grid points = {nx * ny}")
        if missing > 0:
            print(
                f"  WARNING: {missing} grid points are NaN (incomplete grid). "
                "Interpolation accuracy may be reduced."
            )

    return x_grid, y_grid, fes_grid


def load_plumed_scalar_field_2d(
    filename,
    x_col=0,
    y_col=1,
    field_col=2,
    verbose=True,
):
    """
    Load a generic 2D scalar field f(x,y) from a PLUMED-style .dat file.

    This is a thin wrapper around `load_plumed_fes_2d` that does NOT
    subtract the minimum and just renames the last column.

    Parameters
    ----------
    filename : str
        Path to the text file.
    x_col, y_col, field_col : int
        Column indices (0-based) for x, y and f(x,y).
    verbose : bool
        If True, prints basic info about the grid.

    Returns
    -------
    x_grid, y_grid : 1D arrays
        Grid coordinates.
    field_grid     : 2D array, shape (nx, ny)
        Scalar field values on the (x_grid, y_grid) mesh.
    """
    x_grid, y_grid, field_grid = load_plumed_fes_2d(
        filename,
        x_col=x_col,
        y_col=y_col,
        fes_col=field_col,
        subtract_min=False,
        verbose=verbose,
    )
    return x_grid, y_grid, field_grid


def load_diffusion_scalar_2d_from_plumed(
    filename,
    x_col=0,
    y_col=1,
    D_col=2,
    verbose=True,
):
    """
    Convenience wrapper for a position-dependent scalar diffusion
    coefficient D(x,y) stored in a PLUMED-like table.

    Parameters
    ----------
    filename : str
        Path to the text file.
    x_col, y_col, D_col : int
        Column indices (0-based) for x, y and D(x,y).
    verbose : bool
        If True, prints basic info about the grid.

    Returns
    -------
    x_grid, y_grid : 1D arrays
    D_grid         : 2D array, shape (nx, ny)
    """
    return load_plumed_scalar_field_2d(
        filename,
        x_col=x_col,
        y_col=y_col,
        field_col=D_col,
        verbose=verbose,
    )


def load_plumed_fes_1d(
    filename,
    x_col=0,
    fes_col=1,
    subtract_min=True,
    verbose=True,
):
    """Load a 1D free energy surface from a PLUMED-style .dat file.

    Assumes at least two numeric columns (``x``, ``F(x)``) and that
    lines starting with ``'#'`` are comments.

    Parameters
    ----------
    filename : str or path-like
        Path to the PLUMED ``.dat`` file.
    x_col : int
        Column index for the coordinate (default 0).
    fes_col : int
        Column index for the free energy (default 1).
    subtract_min : bool
        If True, shift so the global minimum is zero.
    verbose : bool
        Print a summary after loading.

    Returns
    -------
    x_grid : ndarray, shape (n,)
        Sorted coordinate values.
    fes_grid : ndarray, shape (n,)
        Corresponding free-energy values.
    """
    data = np.loadtxt(filename, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    x = data[:, x_col]
    fes = data[:, fes_col]

    # Sort by x
    idx = np.argsort(x)
    x_grid = np.asarray(x[idx], dtype=float)
    fes_grid = np.asarray(fes[idx], dtype=float)

    if subtract_min:
        fes_grid = fes_grid - np.nanmin(fes_grid)

    if verbose:
        print(
            f"[load_plumed_fes_1d] Loaded 1D FES from '{filename}' "
            f"with {x_grid.size} points."
        )

    return x_grid, fes_grid


class FESPotential:
    """Picklable 2D free-energy-surface potential.

    Given a regular ``(x_grid, y_grid, fes_grid)`` mesh, this class
    builds an interpolating potential callable::

        U, F = potential(np.array([x, y]))

    where ``F = −∇U`` is the 2D force vector.

    Two interpolation back-ends are supported:

    * **``'spline'``** – ``scipy.interpolate.RectBivariateSpline``
      (high accuracy, requires SciPy).
    * **``'bilinear'``** – pure-NumPy bilinear interpolation with
      pre-computed gradient grids (faster for large grids, no SciPy
      needed).
    * **``'auto'``** (default) – selects spline for small grids and
      bilinear for grids with ≥ *auto_bilinear_npoints* points.

    The object is designed to be **picklable** for multiprocessing:
    the SciPy spline is dropped during pickle and rebuilt on unpickle.

    Parameters
    ----------
    x_grid, y_grid : array_like
        Sorted 1D coordinate arrays.
    fes_grid : array_like, shape (len(x_grid), len(y_grid))
        Free-energy values on the regular mesh.
    method : {'spline', 'bilinear', 'auto'}
        Interpolation back-end.
    kx, ky : int
        Spline degrees (default 3).
    s : float
        Spline smoothing factor (default 0).
    extrapolate : bool
        If ``False`` (default), positions outside the grid are
        clamped.
    auto_bilinear_npoints : int
        Grid-size threshold for the ``'auto'`` selector.
    uniform_tol : float
        Tolerance for detecting uniform grid spacing.
    """

    def __init__(
        self,
        x_grid,
        y_grid,
        fes_grid,
        method="spline",
        kx=3,
        ky=3,
        s=0.0,
        extrapolate=False,
        auto_bilinear_npoints: int = 100_000,
        uniform_tol: float = 1e-10,
    ):
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.y_grid = np.asarray(y_grid, dtype=float)
        self.fes_grid = np.asarray(fes_grid, dtype=float)

        if self.fes_grid.shape != (self.x_grid.size, self.y_grid.size):
            raise ValueError(
                f"fes_grid shape {self.fes_grid.shape} incompatible with "
                f"x_grid ({self.x_grid.size}) and y_grid ({self.y_grid.size})."
            )

        # --- grid meta (fast indexing for uniform grids) ---
        self._uniform_tol = float(uniform_tol)
        self._auto_bilinear_npoints = int(auto_bilinear_npoints)

        # Precompute uniform-grid parameters for fast bilinear interpolation (if applicable)
        dx_arr = np.diff(self.x_grid)
        dy_arr = np.diff(self.y_grid)
        self._uniform_x = (dx_arr.size > 0 and np.allclose(dx_arr, dx_arr[0]))
        self._uniform_y = (dy_arr.size > 0 and np.allclose(dy_arr, dy_arr[0]))
        self._x_min = float(self.x_grid[0])
        self._y_min = float(self.y_grid[0])
        self._inv_dx = (1.0 / float(dx_arr[0])) if self._uniform_x else None
        self._inv_dy = (1.0 / float(dy_arr[0])) if self._uniform_y else None

        def _is_uniform_1d(arr: np.ndarray, tol: float):
            if arr.size < 2:
                return True, 0.0
            d = np.diff(arr)
            d0 = float(np.mean(d))
            scale = max(1.0, abs(d0))
            ok = bool(np.all(np.abs(d - d0) <= tol * scale))
            return ok, d0

        self._uniform_x, self._dx = _is_uniform_1d(self.x_grid, self._uniform_tol)
        self._uniform_y, self._dy = _is_uniform_1d(self.y_grid, self._uniform_tol)
        self._x0 = float(self.x_grid[0]) if self.x_grid.size else 0.0
        self._y0 = float(self.y_grid[0]) if self.y_grid.size else 0.0

        # A uniform grid is required for the O(1) index computation path.
        # Also guard against pathological zero spacing.
        self._uniform_grid = bool(
            self._uniform_x and self._uniform_y and (self._dx != 0.0) and (self._dy != 0.0)
        )

        # --- interpolation mode ---
        self.method = method
        self.kx = kx
        self.ky = ky
        self.s = s
        self.extrapolate = extrapolate

        # 'auto' chooses bilinear for large grids to avoid very expensive spline evals
        # in tight inner loops (e.g. BD/Langevin stepping). Users can still force
        # 'spline' explicitly.
        if self.method == "auto":
            npts = int(self.fes_grid.size)
            if (not SCIPY_AVAILABLE) or (npts >= self._auto_bilinear_npoints):
                self.method = "bilinear"
            else:
                self.method = "spline"

        if self.method == "spline" and SCIPY_AVAILABLE:
            self._mode = "spline"
            self._build_spline()
        else:
            if self.method not in ("spline", "bilinear"):
                raise ValueError(
                    f"Unknown FESPotential method={self.method!r}. Use 'spline', 'bilinear', or 'auto'."
                )
            if self.method == "spline" and not SCIPY_AVAILABLE:
                print("Warning: SciPy not available, falling back to bilinear interpolation.")
            self._mode = "bilinear"
            self._build_grad_grid()

    # --- internal builders ---

    def _build_spline(self):
        self.spl = RectBivariateSpline(
            self.x_grid,
            self.y_grid,
            self.fes_grid,
            kx=self.kx,
            ky=self.ky,
            s=self.s,
        )

    def _build_grad_grid(self):
        self.dUdx_grid, self.dUdy_grid = np.gradient(
            self.fes_grid, self.x_grid, self.y_grid, edge_order=2
        )

    # --- helpers ---
    def evaluate_U_on_grid(self, xs, ys):
        """Evaluate U(x,y) on a tensor-product grid (vectorized).

        This avoids calling `__call__` in Python loops and is used to speed up
        basin detection / coarse scans.

        Parameters
        ----------
        xs, ys : array_like
            1D coordinate arrays.

        Returns
        -------
        U : ndarray, shape (len(xs), len(ys))
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1:
            raise ValueError('xs and ys must be 1D arrays')

        if self._mode == 'spline':
            # RectBivariateSpline is already vectorized for grid=True.
            return np.asarray(self.spl(xs, ys, grid=True), dtype=float)

        # Bilinear on a mesh (vectorized).
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        X, Y = self._handle_bounds(X, Y)

        nx = self.x_grid.size
        ny = self.y_grid.size

        i = np.searchsorted(self.x_grid, X) - 1
        j = np.searchsorted(self.y_grid, Y) - 1
        i = np.clip(i, 0, nx - 2)
        j = np.clip(j, 0, ny - 2)

        x0 = self.x_grid[i]
        x1 = self.x_grid[i + 1]
        y0 = self.y_grid[j]
        y1 = self.y_grid[j + 1]

        # Handle degenerate cells robustly (should not happen in regular grids).
        tx = np.where(x1 == x0, 0.0, (X - x0) / (x1 - x0))
        ty = np.where(y1 == y0, 0.0, (Y - y0) / (y1 - y0))

        f00 = self.fes_grid[i, j]
        f10 = self.fes_grid[i + 1, j]
        f01 = self.fes_grid[i, j + 1]
        f11 = self.fes_grid[i + 1, j + 1]

        U = (1.0 - tx) * (1.0 - ty) * f00 + tx * (1.0 - ty) * f10 + (1.0 - tx) * ty * f01 + tx * ty * f11
        return np.asarray(U, dtype=float)

    def coarse_subsample(self, nx, ny, xlim=None, ylim=None):
        """Fast coarse subsampling of the underlying grid (no interpolation).

        Useful for basin detection where you only need U on a coarse grid.

        Returns
        -------
        xs, ys, U : (ndarray, ndarray, ndarray)
            xs has length nx, ys has length ny, U shape (nx, ny).
        """
        nx = int(nx)
        ny = int(ny)
        if nx < 2 or ny < 2:
            raise ValueError('nx, ny must be >= 2')

        xarr = self.x_grid
        yarr = self.y_grid

        # Apply ROI if provided.
        if xlim is None:
            i0, i1 = 0, len(xarr) - 1
        else:
            xlo, xhi = float(xlim[0]), float(xlim[1])
            i0 = int(np.searchsorted(xarr, xlo, side='left'))
            i1 = int(np.searchsorted(xarr, xhi, side='right') - 1)
            i0 = max(0, min(i0, len(xarr) - 2))
            i1 = max(i0 + 1, min(i1, len(xarr) - 1))

        if ylim is None:
            j0, j1 = 0, len(yarr) - 1
        else:
            ylo, yhi = float(ylim[0]), float(ylim[1])
            j0 = int(np.searchsorted(yarr, ylo, side='left'))
            j1 = int(np.searchsorted(yarr, yhi, side='right') - 1)
            j0 = max(0, min(j0, len(yarr) - 2))
            j1 = max(j0 + 1, min(j1, len(yarr) - 1))

        ix = np.linspace(i0, i1, nx, dtype=int)
        iy = np.linspace(j0, j1, ny, dtype=int)
        xs = xarr[ix]
        ys = yarr[iy]
        U = self.fes_grid[np.ix_(ix, iy)]
        return xs, ys, np.asarray(U, dtype=float)

    def _handle_bounds(self, X, Y):
        if self.extrapolate:
            return X, Y
        Xc = np.clip(X, self.x_grid[0], self.x_grid[-1])
        Yc = np.clip(Y, self.y_grid[0], self.y_grid[-1])
        return Xc, Yc

    def _interp_bilinear(self, Z, X, Y):
        X, Y = self._handle_bounds(X, Y)
        nx = self.x_grid.size
        ny = self.y_grid.size

        # Fast path for (nearly) uniform grids.
        if self._uniform_grid:
            i = int(np.floor((X - self._x0) / self._dx))
            j = int(np.floor((Y - self._y0) / self._dy))
            i = 0 if i < 0 else (nx - 2 if i > nx - 2 else i)
            j = 0 if j < 0 else (ny - 2 if j > ny - 2 else j)

            tx = (X - (self._x0 + i * self._dx)) / self._dx
            ty = (Y - (self._y0 + j * self._dy)) / self._dy
        else:
            i = np.searchsorted(self.x_grid, X) - 1
            j = np.searchsorted(self.y_grid, Y) - 1
            i = int(np.clip(i, 0, nx - 2))
            j = int(np.clip(j, 0, ny - 2))

            x0, x1 = self.x_grid[i], self.x_grid[i + 1]
            y0, y1 = self.y_grid[j], self.y_grid[j + 1]

            if x1 == x0 or y1 == y0:
                return float(Z[i, j])  # degenerate cell

            tx = (X - x0) / (x1 - x0)
            ty = (Y - y0) / (y1 - y0)

        z00 = Z[i, j]
        z10 = Z[i + 1, j]
        z01 = Z[i, j + 1]
        z11 = Z[i + 1, j + 1]

        z0 = (1.0 - tx) * z00 + tx * z10
        z1 = (1.0 - tx) * z01 + tx * z11
        z = (1.0 - ty) * z0 + ty * z1
        return float(z)

    def _interp_bilinear_u_and_grad(self, X: float, Y: float):
        """Fast scalar bilinear interpolation for U and its gradient.

        This computes the interpolation weights once and applies them to
        (U, dU/dx, dU/dy) grids. It is a hot path in Langevin/BD loops.
        """
        # Determine cell indices and fractional coordinates
        if self._uniform_x and self._uniform_y:
            nx, ny = self.fes_grid.shape
            i = int((X - self._x_min) * self._inv_dx)
            j = int((Y - self._y_min) * self._inv_dy)
            i = int(np.clip(i, 0, nx - 2))
            j = int(np.clip(j, 0, ny - 2))
            x0 = self._x_min + i / self._inv_dx
            y0 = self._y_min + j / self._inv_dy
            tx = (X - x0) * self._inv_dx
            ty = (Y - y0) * self._inv_dy
        else:
            nx, ny = self.fes_grid.shape
            i = np.searchsorted(self.x_grid, X) - 1
            j = np.searchsorted(self.y_grid, Y) - 1
            i = int(np.clip(i, 0, nx - 2))
            j = int(np.clip(j, 0, ny - 2))
            x0, x1 = self.x_grid[i], self.x_grid[i + 1]
            y0, y1 = self.y_grid[j], self.y_grid[j + 1]
            if x1 == x0 or y1 == y0:
                U = float(self.fes_grid[i, j])
                dUdx = float(self.dUdx_grid[i, j])
                dUdy = float(self.dUdy_grid[i, j])
                return U, dUdx, dUdy
            tx = (X - x0) / (x1 - x0)
            ty = (Y - y0) / (y1 - y0)

        # Weights
        w00 = (1.0 - tx) * (1.0 - ty)
        w10 = tx * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w11 = tx * ty

        i1 = i + 1
        j1 = j + 1

        U = (w00 * self.fes_grid[i, j] + w10 * self.fes_grid[i1, j] +
             w01 * self.fes_grid[i, j1] + w11 * self.fes_grid[i1, j1])
        dUdx = (w00 * self.dUdx_grid[i, j] + w10 * self.dUdx_grid[i1, j] +
                w01 * self.dUdx_grid[i, j1] + w11 * self.dUdx_grid[i1, j1])
        dUdy = (w00 * self.dUdy_grid[i, j] + w10 * self.dUdy_grid[i1, j] +
                w01 * self.dUdy_grid[i, j1] + w11 * self.dUdy_grid[i1, j1])
        return float(U), float(dUdx), float(dUdy)

    # --- main callable ---

    def __call__(self, position):
        """Evaluate the potential at a 2D point.

        Parameters
        ----------
        position : array_like, shape (2,)
            ``[x, y]`` coordinate.

        Returns
        -------
        U : float
            Free energy.
        F : ndarray, shape (2,)
            Force vector :math:`F = -\nabla U`.
        """
        pos = np.asarray(position, dtype=float)
        if pos.shape != (2,):
            raise ValueError("Position must be a 2D vector [x, y].")

        X, Y = pos

        if self._mode == "spline":
            Xc, Yc = self._handle_bounds(X, Y)
            U = float(self.spl.ev(Xc, Yc))
            dUdx = float(self.spl.ev(Xc, Yc, dx=1, dy=0))
            dUdy = float(self.spl.ev(Xc, Yc, dx=0, dy=1))
        else:
            U, dUdx, dUdy = self._interp_bilinear_u_and_grad(X, Y)

        F = -np.array([dUdx, dUdy])
        return U, F

    # --- custom pickling: drop heavy objects, rebuild in child ---

    def __getstate__(self):
        state = self.__dict__.copy()
        if "spl" in state:
            state["spl"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if getattr(self, "_mode", None) == "spline":
            # If SciPy is available, rebuild the spline. Otherwise, degrade
            # gracefully to the bilinear interpolator so pickled objects remain
            # usable in SciPy-less environments (e.g. some MPI/worker nodes).
            if getattr(self, "spl", None) is None and SCIPY_AVAILABLE:
                self._build_spline()
            elif getattr(self, "spl", None) is None and not SCIPY_AVAILABLE:
                self._mode = "bilinear"
                # ensure gradients exist for force evaluation
                if not hasattr(self, "dUdx_grid") or not hasattr(self, "dUdy_grid"):
                    self._build_grad_grid()
        else:
            if not hasattr(self, "dUdx_grid") or not hasattr(self, "dUdy_grid"):
                self._build_grad_grid()

    def plot(self, levels=50, cmap="viridis", title=None):
        """
        Plot the underlying FES grid as a colormap.
        """
        if title is None:
            title = "FESPotential grid"
        plot_fes_colormap(
            self.x_grid,
            self.y_grid,
            self.fes_grid,
            levels=levels,
            cmap=cmap,
            title=title,
        )

    def plot_interpolated(
        self,
        nx=200,
        ny=200,
        levels=50,
        cmap="viridis",
        title=None,
    ):
        xs = np.linspace(self.x_grid[0], self.x_grid[-1], nx)
        ys = np.linspace(self.y_grid[0], self.y_grid[-1], ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = np.zeros_like(X)

        for i in range(nx):
            for j in range(ny):
                Z[i, j], _ = self([X[i, j], Y[i, j]])

        plot_fes_colormap(
            xs,
            ys,
            Z,
            levels=levels,
            cmap=cmap,
            title=title or "Interpolated FES",
        )


class FESPotential1D:
    """Picklable 1D free-energy-surface potential.

    Interpolates ``U(x)`` via ``numpy.interp`` and computes the force
    :math:`F = -dU/dx` via a symmetric finite-difference stencil.

    The callable interface is compatible with the BAOAB integrator::

        U, F = potential(np.array([x]))   # F.shape == (1,)

    Parameters
    ----------
    x_grid : array_like
        Strictly increasing 1D coordinate array.
    fes_grid : array_like
        Free-energy values (same shape as *x_grid*).
    """

    def __init__(self, x_grid, fes_grid):
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.fes_grid = np.asarray(fes_grid, dtype=float)

        if self.x_grid.ndim != 1:
            raise ValueError("x_grid must be 1D for FESPotential1D.")
        if self.fes_grid.shape != self.x_grid.shape:
            raise ValueError(
                f"fes_grid shape {self.fes_grid.shape} must match x_grid "
                f"shape {self.x_grid.shape}."
            )

        dx = np.diff(self.x_grid)
        if np.any(dx <= 0):
            raise ValueError("x_grid must be strictly increasing.")
        self._dx_min = dx.min()

    def _interp_energy(self, x):
        return float(np.interp(x, self.x_grid, self.fes_grid))

    def _interp_force(self, x):
        """
        Finite-difference estimate of -dU/dx at position x.
        """
        # Use a small step based on the grid spacing; clamp to domain
        h = 0.5 * self._dx_min
        xp = min(x + h, self.x_grid[-1])
        xm = max(x - h, self.x_grid[0])

        Up = self._interp_energy(xp)
        Um = self._interp_energy(xm)

        if xp != xm:
            dUdx = (Up - Um) / (xp - xm)
        else:
            dUdx = 0.0

        return np.array([-dUdx], dtype=float)

    def __call__(self, position):
        """Evaluate the 1D potential.

        Parameters
        ----------
        position : scalar or array_like
            1D coordinate ``[x]``.

        Returns
        -------
        U : float
            Free energy.
        F : ndarray, shape (1,)
            Force :math:`F = -dU/dx`.
        """
        pos = np.asarray(position, dtype=float).ravel()
        if pos.size != 1:
            raise ValueError("FESPotential1D expects a 1D position [x].")

        x = float(pos[0])
        U = self._interp_energy(x)
        F = self._interp_force(x)
        return U, F

    def plot(self, ax=None, xlabel="x", ylabel="F(x)", title="1D FES"):
        """
        Quick 1D line plot of the underlying FES.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.x_grid, self.fes_grid)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if ax is not None and ax.figure is not None:
            ax.figure.tight_layout()


def make_fes_potential_from_grid(
    x_grid,
    y_grid,
    fes_grid,
    method="auto",
    kx=3,
    ky=3,
    s=0.0,
    extrapolate=False,
):
    """Build a picklable :class:`FESPotential` from grid arrays.

    Parameters
    ----------
    x_grid, y_grid : array_like
        1D coordinate arrays.
    fes_grid : array_like, shape (nx, ny)
        Free-energy values.
    method : {'spline', 'bilinear', 'auto'}
        Interpolation back-end.
    kx, ky : int
        Spline degrees.
    s : float
        Smoothing.
    extrapolate : bool
        Allow extrapolation outside the grid.

    Returns
    -------
    FESPotential
    """
    return FESPotential(
        x_grid,
        y_grid,
        fes_grid,
        method=method,
        kx=kx,
        ky=ky,
        s=s,
        extrapolate=extrapolate,
    )


def make_fes_potential_from_plumed(
    filename,
    x_col=0,
    y_col=1,
    fes_col=2,
    subtract_min=True,
    method="auto",
    kx=3,
    ky=3,
    s=0.0,
    extrapolate=False,
    verbose=True,
    plot=True,
    plot_nx=200,
    plot_ny=200,
    plot_levels=50,
    cmap="viridis",
):
    """Build a :class:`FESPotential` directly from a PLUMED FES file.

    Combines :func:`load_plumed_fes_2d` and
    :func:`make_fes_potential_from_grid` in a single call.
    Optionally plots the interpolated FES on a refined grid.

    Parameters
    ----------
    filename : str
        Path to the PLUMED ``.dat`` file.
    x_col, y_col, fes_col : int
        Column indices.
    subtract_min : bool
        Shift FES minimum to zero.
    method : str
        Interpolation back-end.
    kx, ky : int
        Spline degrees.
    s : float
        Smoothing.
    extrapolate : bool
        Allow extrapolation.
    verbose : bool
        Print grid info.
    plot : bool
        Show a diagnostic colormap.
    plot_nx, plot_ny : int
        Refined grid size for plotting.
    plot_levels : int
        Contour levels.
    cmap : str
        Matplotlib colormap.

    Returns
    -------
    FESPotential
    """
    x_grid, y_grid, fes_grid = load_plumed_fes_2d(
        filename,
        x_col=x_col,
        y_col=y_col,
        fes_col=fes_col,
        subtract_min=subtract_min,
        verbose=verbose,
    )

    fes_pot = make_fes_potential_from_grid(
        x_grid,
        y_grid,
        fes_grid,
        method=method,
        kx=kx,
        ky=ky,
        s=s,
        extrapolate=extrapolate,
    )

    if plot:
        xs = np.linspace(x_grid[0], x_grid[-1], plot_nx)
        ys = np.linspace(y_grid[0], y_grid[-1], plot_ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = np.zeros_like(X)

        for i in range(plot_nx):
            for j in range(plot_ny):
                Z[i, j], _ = fes_pot([X[i, j], Y[i, j]])

        title = f"Interpolated FES from {os.path.basename(filename)}"
        plot_fes_colormap(
            xs,
            ys,
            Z,
            levels=plot_levels,
            cmap=cmap,
            title=title,
        )

    return fes_pot

def make_fes_potential_from_plumed_1d(
    filename,
    x_col=0,
    fes_col=1,
    subtract_min=True,
    verbose=True,
):
    """Build a :class:`FESPotential1D` from a PLUMED 1D FES file.

    Parameters
    ----------
    filename : str
        Path to the PLUMED ``.dat`` file.
    x_col, fes_col : int
        Column indices.
    subtract_min : bool
        Shift FES minimum to zero.
    verbose : bool
        Print grid info.

    Returns
    -------
    FESPotential1D
    """
    x_grid, fes_grid = load_plumed_fes_1d(
        filename,
        x_col=x_col,
        fes_col=fes_col,
        subtract_min=subtract_min,
        verbose=verbose,
    )
    return FESPotential1D(x_grid, fes_grid)

def plot_fes_colormap(
    x_grid,
    y_grid,
    fes_grid,
    levels=50,
    cmap="viridis",
    title=None,
):
    """Plot a 2D FES as a filled-contour colormap.

    Parameters
    ----------
    x_grid, y_grid : array_like
        1D coordinate arrays.
    fes_grid : array_like, shape (nx, ny)
        Free-energy values (NaNs are masked).
    levels : int
        Number of contour levels.
    cmap : str
        Matplotlib colormap name.
    title : str or None
        Figure title.
    """
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    Z = np.ma.masked_invalid(fes_grid)

    plt.figure(figsize=(6, 5))
    cf = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
    cbar = plt.colorbar(cf)
    cbar.set_label("FES")

    plt.xlabel("CV1")
    plt.ylabel("CV2")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()
