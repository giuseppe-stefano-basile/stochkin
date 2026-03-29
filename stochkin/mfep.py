"""
MFEP utilities for 2D free-energy surfaces.

This submodule provides:
  1) Grid-based MFEP search (Dijkstra/minimax)
  2) NEB-style path refinement
  3) Export of a 1D profile s -> F(s) compatible with 1D CTMC workflows
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import heapq

import numpy as np

from scipy.interpolate import RectBivariateSpline as _RBS

from .fes import load_plumed_fes_2d


def _as_xy(point: Sequence[float], name: str) -> Tuple[float, float]:
    arr = np.asarray(point, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two values [x, y].")
    return float(arr[0]), float(arr[1])


def _cumulative_arclength(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if x.size == 0:
        return np.asarray([], dtype=float)
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.zeros(x.size, dtype=float)
    if ds.size:
        s[1:] = np.cumsum(ds)
    return s


def _resample_polyline(path_xy: np.ndarray, n_points: int) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise ValueError("path_xy must have shape (N, 2).")
    if path_xy.shape[0] < 2:
        raise ValueError("path_xy must contain at least two points.")
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")

    s = _cumulative_arclength(path_xy[:, 0], path_xy[:, 1])
    total = float(s[-1])
    if total <= 0.0:
        out = np.repeat(path_xy[:1], n_points, axis=0)
        out[-1] = path_xy[-1]
        return out

    st = np.linspace(0.0, total, int(n_points))
    xr = np.interp(st, s, path_xy[:, 0])
    yr = np.interp(st, s, path_xy[:, 1])
    return np.column_stack([xr, yr])


@dataclass
class MFEPPath:
    """
    MFEP trajectory and projected 1D free-energy profile.

    Attributes
    ----------
    x, y : 1D arrays
        MFEP coordinates in the original CV space.
    s : 1D array
        Curvilinear coordinate along the path (same length as x/y).
    F : 1D array
        Free energy sampled along the path (same length as x/y).
    method : str
        Path construction method, e.g. "grid" or "neb".
    objective : str
        Optimization objective, e.g. "integral", "barrier", "refined".
    indices : list[(ix,iy)] or None
        Optional grid indices (available for grid paths).
    metadata : dict
        Additional run details.
    """

    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    F: np.ndarray
    method: str
    objective: str
    indices: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        self.s = np.asarray(self.s, dtype=float).ravel()
        self.F = np.asarray(self.F, dtype=float).ravel()

        n = self.x.size
        if n == 0:
            raise ValueError("MFEPPath is empty.")
        if self.y.size != n or self.s.size != n or self.F.size != n:
            raise ValueError("x, y, s, F must have the same length.")

    def as_xy(self) -> np.ndarray:
        """Return path as shape (N,2) array."""
        return np.column_stack([self.x, self.y])

    def save_profile_1d(
        self,
        filename: str | Path,
        *,
        subtract_min: bool = True,
        fmt: str = "%.10f",
    ) -> Path:
        """
        Save 1D profile for CTMC scripts as two columns: s, F.

        Output format is PLUMED-like and compatible with loaders that read
        the first two numeric columns as (coordinate, free energy).
        """
        out = Path(filename).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        f = self.F - np.nanmin(self.F) if subtract_min else np.asarray(self.F, dtype=float)
        arr = np.column_stack([self.s, f])
        header = (
            f"# s F | method={self.method} objective={self.objective} "
            f"| n={self.s.size} | subtract_min={subtract_min}\n# s F"
        )
        np.savetxt(out, arr, fmt=fmt, header=header)
        return out

    def save_path_xyf(
        self,
        filename: str | Path,
        *,
        subtract_min: bool = False,
        fmt: str = "%.10f",
    ) -> Path:
        """Save full path as four columns: s, x, y, F."""
        out = Path(filename).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        f = self.F - np.nanmin(self.F) if subtract_min else np.asarray(self.F, dtype=float)
        arr = np.column_stack([self.s, self.x, self.y, f])
        header = (
            f"# s x y F | method={self.method} objective={self.objective} "
            f"| n={self.s.size} | subtract_min={subtract_min}"
        )
        np.savetxt(out, arr, fmt=fmt, header=header)
        return out


class GridMFEP:
    """
    Grid-based MFEP finder on a 2D FES.

    Supports two objectives:
      - "integral": minimum integrated free energy along the path
      - "barrier": minimax barrier path (minimum maximum encountered F)
    """

    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, fes_grid: np.ndarray):
        self.x = np.asarray(x_grid, dtype=float).ravel()
        self.y = np.asarray(y_grid, dtype=float).ravel()
        self.F = np.asarray(fes_grid, dtype=float)

        if self.x.size < 2 or self.y.size < 2:
            raise ValueError("x_grid and y_grid must contain at least 2 points each.")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x_grid must be strictly increasing.")
        if not np.all(np.diff(self.y) > 0):
            raise ValueError("y_grid must be strictly increasing.")
        if self.F.shape != (self.x.size, self.y.size):
            raise ValueError(
                f"fes_grid shape mismatch: expected {(self.x.size, self.y.size)}, got {self.F.shape}"
            )

        self._finite_mask = np.isfinite(self.F)
        if not np.any(self._finite_mask):
            raise ValueError("fes_grid contains no finite values.")

        self._fmin = float(np.nanmin(self.F))
        self._F_cost = np.where(self._finite_mask, self.F - self._fmin, np.nan)

    @classmethod
    def from_plumed(
        cls,
        filename: str | Path,
        *,
        x_col: int = 0,
        y_col: int = 1,
        fes_col: int = 2,
        subtract_min: bool = True,
        verbose: bool = False,
    ) -> "GridMFEP":
        x, y, F = load_plumed_fes_2d(
            str(filename),
            x_col=x_col,
            y_col=y_col,
            fes_col=fes_col,
            subtract_min=subtract_min,
            verbose=verbose,
        )
        return cls(x, y, F)

    def coordinate_to_index(self, point: Sequence[float]) -> Tuple[int, int]:
        x0, y0 = _as_xy(point, "point")
        ix = int(np.argmin(np.abs(self.x - x0)))
        iy = int(np.argmin(np.abs(self.y - y0)))
        return ix, iy

    def _nearest_finite_index(self, idx: Tuple[int, int]) -> Tuple[int, int]:
        ix, iy = int(idx[0]), int(idx[1])
        if self._finite_mask[ix, iy]:
            return ix, iy
        finite_idx = np.argwhere(self._finite_mask)
        if finite_idx.size == 0:
            raise ValueError("No finite cells available in FES grid.")
        dx = self.x[finite_idx[:, 0]] - self.x[ix]
        dy = self.y[finite_idx[:, 1]] - self.y[iy]
        k = int(np.argmin(dx * dx + dy * dy))
        return int(finite_idx[k, 0]), int(finite_idx[k, 1])

    def _neighbors(self, ix: int, iy: int, connectivity: int):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        nx, ny = self.F.shape
        for dxi, dyi in steps:
            jx = ix + dxi
            jy = iy + dyi
            if 0 <= jx < nx and 0 <= jy < ny:
                ds = float(np.hypot(self.x[jx] - self.x[ix], self.y[jy] - self.y[iy]))
                yield jx, jy, ds

    def _allowed(
        self,
        ix: int,
        iy: int,
        *,
        f_threshold: Optional[float],
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> bool:
        if not self._finite_mask[ix, iy]:
            return False
        if f_threshold is None:
            return True
        if (ix, iy) == start or (ix, iy) == end:
            return True
        return bool(self.F[ix, iy] <= float(f_threshold))

    def _reconstruct_path(
        self,
        parent: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [end]
        cur = end
        while cur != start:
            px, py = parent[cur[0], cur[1]]
            if px < 0 or py < 0:
                raise RuntimeError("Path reconstruction failed (disconnected graph).")
            cur = (int(px), int(py))
            path.append(cur)
        path.reverse()
        return path

    def _build_result(
        self,
        indices: List[Tuple[int, int]],
        *,
        method: str,
        objective: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MFEPPath:
        xs = np.asarray([self.x[i] for i, _ in indices], dtype=float)
        ys = np.asarray([self.y[j] for _, j in indices], dtype=float)
        fs = np.asarray([self.F[i, j] for i, j in indices], dtype=float)
        ss = _cumulative_arclength(xs, ys)
        return MFEPPath(
            x=xs,
            y=ys,
            s=ss,
            F=fs,
            method=method,
            objective=objective,
            indices=list(indices),
            metadata=dict(metadata or {}),
        )

    def find_path(
        self,
        start: Sequence[float],
        end: Sequence[float],
        *,
        objective: str = "integral",
        connectivity: int = 8,
        f_threshold: Optional[float] = None,
        project_to_finite: bool = True,
    ) -> MFEPPath:
        """
        Compute a grid MFEP between start and end points.

        Parameters
        ----------
        start, end : sequence[float]
            Start/end coordinates [x, y].
        objective : {"integral", "barrier"}
            integral -> minimum integrated F(s)
            barrier  -> minimax path (minimum maximum F)
        connectivity : {4, 8}
            Grid connectivity.
        f_threshold : float or None
            Optional hard mask: allow only cells with F <= f_threshold
            (start/end are always allowed).
        project_to_finite : bool
            If True, snap start/end to nearest finite cell if needed.
        """
        objective = str(objective).strip().lower()
        if objective not in ("integral", "barrier"):
            raise ValueError("objective must be 'integral' or 'barrier'.")

        start_idx = self.coordinate_to_index(start)
        end_idx = self.coordinate_to_index(end)
        if project_to_finite:
            start_idx = self._nearest_finite_index(start_idx)
            end_idx = self._nearest_finite_index(end_idx)

        if not self._allowed(*start_idx, f_threshold=f_threshold, start=start_idx, end=end_idx):
            raise ValueError("Start point is not on an allowed finite cell.")
        if not self._allowed(*end_idx, f_threshold=f_threshold, start=start_idx, end=end_idx):
            raise ValueError("End point is not on an allowed finite cell.")

        nx, ny = self.F.shape
        parent = np.full((nx, ny, 2), -1, dtype=int)

        if objective == "integral":
            dist = np.full((nx, ny), np.inf, dtype=float)
            sx, sy = start_idx
            dist[sx, sy] = 0.0
            heap = [(0.0, sx, sy)]

            while heap:
                cur_d, ix, iy = heapq.heappop(heap)
                if cur_d > dist[ix, iy]:
                    continue
                if (ix, iy) == end_idx:
                    break
                fi = self._F_cost[ix, iy]
                for jx, jy, ds in self._neighbors(ix, iy, connectivity):
                    if not self._allowed(
                        jx, jy, f_threshold=f_threshold, start=start_idx, end=end_idx
                    ):
                        continue
                    fj = self._F_cost[jx, jy]
                    w = max(0.0, 0.5 * (fi + fj)) * ds
                    nd = cur_d + w
                    if nd < dist[jx, jy]:
                        dist[jx, jy] = nd
                        parent[jx, jy] = (ix, iy)
                        heapq.heappush(heap, (nd, jx, jy))

            if not np.isfinite(dist[end_idx[0], end_idx[1]]):
                raise RuntimeError("No path found under current constraints.")

            path_idx = self._reconstruct_path(parent, start_idx, end_idx)
            return self._build_result(
                path_idx,
                method="grid",
                objective="integral",
                metadata={
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "connectivity": int(connectivity),
                    "f_threshold": None if f_threshold is None else float(f_threshold),
                    "cost_integral": float(dist[end_idx[0], end_idx[1]]),
                },
            )

        # barrier objective: lexicographic (barrier, integral)
        best_barrier = np.full((nx, ny), np.inf, dtype=float)
        best_integral = np.full((nx, ny), np.inf, dtype=float)
        sx, sy = start_idx
        best_barrier[sx, sy] = float(self._F_cost[sx, sy])
        best_integral[sx, sy] = 0.0
        heap = [(best_barrier[sx, sy], 0.0, sx, sy)]
        tol = 1e-15

        while heap:
            cur_b, cur_i, ix, iy = heapq.heappop(heap)
            if cur_b > best_barrier[ix, iy] + tol:
                continue
            if abs(cur_b - best_barrier[ix, iy]) <= tol and cur_i > best_integral[ix, iy] + tol:
                continue
            if (ix, iy) == end_idx:
                break
            fi = self._F_cost[ix, iy]
            for jx, jy, ds in self._neighbors(ix, iy, connectivity):
                if not self._allowed(
                    jx, jy, f_threshold=f_threshold, start=start_idx, end=end_idx
                ):
                    continue
                fj = self._F_cost[jx, jy]
                edge_b = max(fi, fj)
                cand_b = max(cur_b, edge_b)
                cand_i = cur_i + max(0.0, 0.5 * (fi + fj)) * ds
                better = (
                    cand_b < best_barrier[jx, jy] - tol
                    or (
                        abs(cand_b - best_barrier[jx, jy]) <= tol
                        and cand_i < best_integral[jx, jy] - tol
                    )
                )
                if better:
                    best_barrier[jx, jy] = cand_b
                    best_integral[jx, jy] = cand_i
                    parent[jx, jy] = (ix, iy)
                    heapq.heappush(heap, (cand_b, cand_i, jx, jy))

        if not np.isfinite(best_barrier[end_idx[0], end_idx[1]]):
            raise RuntimeError("No path found under current constraints.")

        path_idx = self._reconstruct_path(parent, start_idx, end_idx)
        return self._build_result(
            path_idx,
            method="grid",
            objective="barrier",
            metadata={
                "start_idx": start_idx,
                "end_idx": end_idx,
                "connectivity": int(connectivity),
                "f_threshold": None if f_threshold is None else float(f_threshold),
                "cost_barrier": float(best_barrier[end_idx[0], end_idx[1]] + self._fmin),
                "cost_integral_tiebreak": float(best_integral[end_idx[0], end_idx[1]]),
            },
        )


class NEBMFEP:
    """
    NEB-style path refiner on a 2D FES grid.

    Intended as a refinement step starting from a grid MFEP.
    """

    def __init__(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        fes_grid: np.ndarray,
        *,
        fill_value: Optional[float] = None,
    ):
        self.x = np.asarray(x_grid, dtype=float).ravel()
        self.y = np.asarray(y_grid, dtype=float).ravel()
        self.F = np.asarray(fes_grid, dtype=float)

        if self.x.size < 2 or self.y.size < 2:
            raise ValueError("x_grid and y_grid must contain at least 2 points each.")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x_grid must be strictly increasing.")
        if not np.all(np.diff(self.y) > 0):
            raise ValueError("y_grid must be strictly increasing.")
        if self.F.shape != (self.x.size, self.y.size):
            raise ValueError(
                f"fes_grid shape mismatch: expected {(self.x.size, self.y.size)}, got {self.F.shape}"
            )

        finite = np.isfinite(self.F)
        if not np.any(finite):
            raise ValueError("fes_grid contains no finite values.")
        fmin = float(np.nanmin(self.F))
        fmax = float(np.nanmax(self.F))
        if fill_value is None:
            span = max(1.0, fmax - fmin)
            fill_value = fmax + 0.25 * span
        self.fill_value = float(fill_value)

        self.F_work = np.where(finite, self.F, self.fill_value)
        edge_order = 2 if min(self.F_work.shape) >= 3 else 1
        self.dFdx, self.dFdy = np.gradient(
            self.F_work, self.x, self.y, edge_order=edge_order
        )

        # Smooth bicubic spline for gradient evaluation (C² continuous)
        self._spl = _RBS(self.x, self.y, self.F_work, kx=3, ky=3)

    def _interp_scalar(self, field: np.ndarray, xq: float, yq: float) -> float:
        xq = float(np.clip(xq, self.x[0], self.x[-1]))
        yq = float(np.clip(yq, self.y[0], self.y[-1]))

        ix = int(np.searchsorted(self.x, xq) - 1)
        iy = int(np.searchsorted(self.y, yq) - 1)
        ix = int(np.clip(ix, 0, self.x.size - 2))
        iy = int(np.clip(iy, 0, self.y.size - 2))

        x0, x1 = self.x[ix], self.x[ix + 1]
        y0, y1 = self.y[iy], self.y[iy + 1]
        tx = 0.0 if x1 == x0 else (xq - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (yq - y0) / (y1 - y0)

        f00 = field[ix, iy]
        f10 = field[ix + 1, iy]
        f01 = field[ix, iy + 1]
        f11 = field[ix + 1, iy + 1]

        return float(
            (1.0 - tx) * (1.0 - ty) * f00
            + tx * (1.0 - ty) * f10
            + (1.0 - tx) * ty * f01
            + tx * ty * f11
        )

    def _interp_grad(self, xq: float, yq: float) -> np.ndarray:
        """Gradient via bicubic spline – C² smooth, no grid-cell kinks."""
        xq = float(np.clip(xq, self.x[0], self.x[-1]))
        yq = float(np.clip(yq, self.y[0], self.y[-1]))
        gx = float(self._spl(xq, yq, dx=1, grid=False))
        gy = float(self._spl(xq, yq, dy=1, grid=False))
        return np.asarray([gx, gy], dtype=float)

    def refine(
        self,
        initial_path: np.ndarray,
        *,
        n_images: int = 120,
        k_spring: float = 1.0,
        step_size: float = 0.01,
        max_iter: int = 3000,
        tol: float = 5.0,
        smooth: float = 0.0,
        clip_to_bounds: bool = True,
        max_step: Optional[float] = None,
        adaptive_step: bool = True,
        step_shrink: float = 0.5,
        step_grow: float = 1.02,
        step_min: Optional[float] = None,
        reparam_every: int = 10,
    ) -> MFEPPath:
        """
        Refine an initial path with a simple NEB-style iterative scheme.
        """
        if n_images < 3:
            raise ValueError("n_images must be >= 3.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0.")
        if not (0.0 <= smooth <= 1.0):
            raise ValueError("smooth must be in [0, 1].")
        if max_step is not None and float(max_step) <= 0.0:
            raise ValueError("max_step must be > 0 when provided.")
        if step_min is not None and float(step_min) <= 0.0:
            raise ValueError("step_min must be > 0 when provided.")
        if not (0.0 < float(step_shrink) < 1.0):
            raise ValueError("step_shrink must be in (0,1).")
        if float(step_grow) < 1.0:
            raise ValueError("step_grow must be >= 1.")
        if int(reparam_every) < 0:
            raise ValueError("reparam_every must be >= 0.")

        p0 = np.asarray(initial_path, dtype=float)
        if p0.ndim != 2 or p0.shape[1] != 2:
            raise ValueError("initial_path must have shape (N,2).")
        if p0.shape[0] < 2:
            raise ValueError("initial_path must contain at least two points.")

        images = _resample_polyline(p0, int(n_images))
        images[0] = p0[0]
        images[-1] = p0[-1]

        dx = float(np.median(np.diff(self.x)))
        dy = float(np.median(np.diff(self.y)))
        grid_step = float(np.hypot(dx, dy))
        max_step_eff = float(max_step) if max_step is not None else 3.0 * grid_step
        alpha = float(step_size)
        alpha_max = float(step_size)
        alpha_min = float(step_min) if step_min is not None else max(1e-6, 1e-3 * alpha_max)

        converged = False
        max_force = np.inf
        n_done = 0
        prev_max_force = np.inf

        for it in range(int(max_iter)):
            forces = np.zeros_like(images)
            max_force = 0.0
            for i in range(1, images.shape[0] - 1):
                t = images[i + 1] - images[i - 1]
                nt = float(np.linalg.norm(t))
                if nt <= 1e-15:
                    t = np.asarray([1.0, 0.0], dtype=float)
                else:
                    t = t / nt

                grad = self._interp_grad(images[i, 0], images[i, 1])
                true_force = -grad
                true_perp = true_force - np.dot(true_force, t) * t

                d_f = float(np.linalg.norm(images[i + 1] - images[i]))
                d_b = float(np.linalg.norm(images[i] - images[i - 1]))
                spring = float(k_spring) * (d_f - d_b) * t

                f_tot = true_perp + spring
                forces[i] = f_tot
                max_force = max(max_force, float(np.linalg.norm(f_tot)))

                n_done = it + 1
            if max_force < float(tol):
                converged = True
                break

            if adaptive_step and np.isfinite(prev_max_force):
                if max_force > 1.05 * prev_max_force:
                    alpha = max(alpha * float(step_shrink), alpha_min)
                elif max_force < 0.98 * prev_max_force:
                    alpha = min(alpha * float(step_grow), alpha_max)

            disp = alpha * forces
            if max_step_eff > 0.0:
                dn = np.linalg.norm(disp[1:-1], axis=1)
                scale = np.ones_like(dn)
                m = dn > max_step_eff
                scale[m] = max_step_eff / (dn[m] + 1e-15)
                disp[1:-1] *= scale[:, None]

            images[1:-1] += disp[1:-1]

            if smooth > 0.0:
                images[1:-1] = (
                    (1.0 - float(smooth)) * images[1:-1]
                    + 0.5 * float(smooth) * (images[:-2] + images[2:])
                )

            if int(reparam_every) > 0 and ((it + 1) % int(reparam_every) == 0):
                images = _resample_polyline(images, int(n_images))

            if clip_to_bounds:
                images[:, 0] = np.clip(images[:, 0], self.x[0], self.x[-1])
                images[:, 1] = np.clip(images[:, 1], self.y[0], self.y[-1])

            images[0] = p0[0]
            images[-1] = p0[-1]
            prev_max_force = max_force

        f_path = np.asarray(
            [self._interp_scalar(self.F_work, px, py) for px, py in images], dtype=float
        )
        s_path = _cumulative_arclength(images[:, 0], images[:, 1])
        return MFEPPath(
            x=images[:, 0],
            y=images[:, 1],
            s=s_path,
            F=f_path,
            method="neb",
            objective="refined",
            indices=None,
            metadata={
                "n_images": int(n_images),
                "k_spring": float(k_spring),
                "step_size": float(step_size),
                "max_iter": int(max_iter),
                "tol": float(tol),
                "smooth": float(smooth),
                "clip_to_bounds": bool(clip_to_bounds),
                "adaptive_step": bool(adaptive_step),
                "step_shrink": float(step_shrink),
                "step_grow": float(step_grow),
                "alpha_final": float(alpha),
                "alpha_min": float(alpha_min),
                "alpha_max": float(alpha_max),
                "max_step": float(max_step_eff),
                "reparam_every": int(reparam_every),
                "converged": bool(converged),
                "n_iter": int(n_done),
                "final_max_force": float(max_force),
            },
        )

    def refine_between(
        self,
        start: Sequence[float],
        end: Sequence[float],
        *,
        initializer: Optional[GridMFEP] = None,
        initializer_objective: str = "integral",
        initializer_connectivity: int = 8,
        initializer_f_threshold: Optional[float] = None,
        **neb_kwargs,
    ) -> MFEPPath:
        """
        Build an initial grid path between start/end, then refine with NEB.
        """
        if initializer is None:
            initializer = GridMFEP(self.x, self.y, self.F)
        coarse = initializer.find_path(
            start,
            end,
            objective=initializer_objective,
            connectivity=initializer_connectivity,
            f_threshold=initializer_f_threshold,
            project_to_finite=True,
        )
        init_xy = coarse.as_xy()
        refined = self.refine(init_xy, **neb_kwargs)
        refined.metadata["initializer_method"] = coarse.method
        refined.metadata["initializer_objective"] = coarse.objective
        refined.metadata["initializer_points"] = int(coarse.s.size)
        return refined


    def refine_fire(
        self,
        initial_path: np.ndarray,
        *,
        n_images: int = 120,
        k_spring: float = 1.0,
        dt_start: float = 0.005,
        dt_max: float = 0.05,
        max_iter: int = 5000,
        tol: float = 5.0,
        n_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        alpha_start: float = 0.1,
        f_alpha: float = 0.99,
        smooth: float = 0.0,
        clip_to_bounds: bool = True,
        reparam_every: int = 200,
    ) -> MFEPPath:
        """Refine an NEB path using the FIRE algorithm (Bitzek et al. 2006).

        FIRE (Fast Inertial Relaxation Engine) uses velocity-based
        dynamics with adaptive time-stepping. It converges much faster
        than steepest descent for NEB optimisation.

        Parameters
        ----------
        initial_path : (N, 2) array
        n_images, k_spring : NEB parameters
        dt_start : initial time step
        dt_max : maximum allowed time step
        max_iter : iteration cap
        tol : force convergence tolerance (kJ mol⁻¹ per CV-unit)
        n_min : FIRE delay before acceleration
        f_inc, f_dec : time-step scale factors
        alpha_start, f_alpha : FIRE mixing parameters
        smooth : smoothing factor ∈ [0, 1]
        clip_to_bounds : clip images to grid domain
        reparam_every : reparametrize path every N steps
        """
        p0 = np.asarray(initial_path, dtype=float)
        images = _resample_polyline(p0, int(n_images))
        images[0] = p0[0]
        images[-1] = p0[-1]

        dx = float(np.median(np.diff(self.x)))
        dy = float(np.median(np.diff(self.y)))
        grid_step = float(np.hypot(dx, dy))
        max_step_eff = 2.0 * grid_step  # limit displacement to ~2 grid cells

        dt = float(dt_start)
        alpha = float(alpha_start)
        v = np.zeros_like(images)
        n_pos = 0  # steps since last negative P

        converged = False
        max_force = np.inf
        n_done = 0

        for it in range(int(max_iter)):
            # --- Compute NEB forces on interior images ---
            forces = np.zeros_like(images)
            max_force = 0.0
            for i in range(1, images.shape[0] - 1):
                t = images[i + 1] - images[i - 1]
                nt = float(np.linalg.norm(t))
                if nt <= 1e-15:
                    t = np.asarray([1.0, 0.0], dtype=float)
                else:
                    t = t / nt

                grad = self._interp_grad(images[i, 0], images[i, 1])
                true_force = -grad
                true_perp = true_force - np.dot(true_force, t) * t

                d_f = float(np.linalg.norm(images[i + 1] - images[i]))
                d_b = float(np.linalg.norm(images[i] - images[i - 1]))
                spring = float(k_spring) * (d_f - d_b) * t

                f_tot = true_perp + spring
                forces[i] = f_tot
                max_force = max(max_force, float(np.linalg.norm(f_tot)))

            n_done = it + 1
            if max_force < float(tol):
                converged = True
                break

            # --- FIRE dynamics ---
            P = float(np.sum(v * forces))  # power

            if P > 0.0:
                v_norm = float(np.linalg.norm(v))
                f_norm = float(np.linalg.norm(forces))
                if f_norm > 1e-30:
                    v = (1.0 - alpha) * v + alpha * (v_norm / f_norm) * forces
                n_pos += 1
                if n_pos > n_min:
                    dt = min(dt * f_inc, dt_max)
                    alpha = alpha * f_alpha
            else:
                v[:] = 0.0
                dt = dt * f_dec
                alpha = float(alpha_start)
                n_pos = 0

            # velocity Verlet half-step: v += 0.5*dt*F, x += dt*v
            v += 0.5 * dt * forces
            disp = dt * v[1:-1]
            # Clamp per-image displacement to max_step
            dn = np.linalg.norm(disp, axis=1)
            mask = dn > max_step_eff
            if np.any(mask):
                disp[mask] *= (max_step_eff / (dn[mask, None] + 1e-15))
                # Also limit velocity to prevent further overshoot
                v[1:-1][mask] *= (max_step_eff / (dn[mask, None] + 1e-15))
            images[1:-1] += disp

            if smooth > 0.0:
                images[1:-1] = (
                    (1.0 - float(smooth)) * images[1:-1]
                    + 0.5 * float(smooth) * (images[:-2] + images[2:])
                )

            if int(reparam_every) > 0 and ((it + 1) % int(reparam_every) == 0):
                images = _resample_polyline(images, int(n_images))
                v = np.zeros_like(images)
                n_pos = 0
                alpha = float(alpha_start)

            if clip_to_bounds:
                images[:, 0] = np.clip(images[:, 0], self.x[0], self.x[-1])
                images[:, 1] = np.clip(images[:, 1], self.y[0], self.y[-1])

            images[0] = p0[0]
            images[-1] = p0[-1]

        f_path = np.asarray(
            [self._interp_scalar(self.F_work, px, py) for px, py in images],
            dtype=float,
        )
        s_path = _cumulative_arclength(images[:, 0], images[:, 1])
        return MFEPPath(
            x=images[:, 0],
            y=images[:, 1],
            s=s_path,
            F=f_path,
            method="neb-fire",
            objective="refined",
            indices=None,
            metadata={
                "n_images": int(n_images),
                "k_spring": float(k_spring),
                "dt_start": float(dt_start),
                "dt_max": float(dt_max),
                "max_iter": int(max_iter),
                "tol": float(tol),
                "converged": bool(converged),
                "n_iter": int(n_done),
                "final_max_force": float(max_force),
            },
        )


def compute_mfep_profile_1d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    fes_grid: np.ndarray,
    start: Sequence[float],
    end: Sequence[float],
    *,
    objective: str = "integral",
    connectivity: int = 8,
    f_threshold: Optional[float] = None,
    use_neb: bool = True,
    neb_images: int = 120,
    neb_k_spring: float = 1.0,
    neb_step_size: float = 0.01,
    neb_max_iter: int = 3000,
    neb_tol: float = 5.0,
    neb_smooth: float = 0.0,
    neb_max_step: Optional[float] = None,
    neb_adaptive_step: bool = True,
    neb_reparam_every: Optional[int] = None,
    neb_optimizer: str = "sd",
) -> MFEPPath:
    """
    Convenience wrapper:
      1) grid MFEP between start/end
      2) optional NEB refinement
      3) return MFEPPath containing s,F(s)
    """
    grid = GridMFEP(x_grid, y_grid, fes_grid)
    coarse = grid.find_path(
        start,
        end,
        objective=objective,
        connectivity=connectivity,
        f_threshold=f_threshold,
        project_to_finite=True,
    )
    if not use_neb:
        return coarse

    neb = NEBMFEP(x_grid, y_grid, fes_grid)
    reparam = neb_reparam_every if neb_reparam_every is not None else (200 if neb_optimizer == "fire" else 10)
    if neb_optimizer == "fire":
        refined = neb.refine_fire(
            coarse.as_xy(),
            n_images=neb_images,
            k_spring=neb_k_spring,
            dt_start=neb_step_size,
            max_iter=neb_max_iter,
            tol=neb_tol,
            smooth=neb_smooth,
            clip_to_bounds=True,
            reparam_every=reparam,
        )
    else:
        refined = neb.refine(
            coarse.as_xy(),
            n_images=neb_images,
            k_spring=neb_k_spring,
            step_size=neb_step_size,
            max_iter=neb_max_iter,
            tol=neb_tol,
            smooth=neb_smooth,
            clip_to_bounds=True,
            max_step=neb_max_step,
            adaptive_step=neb_adaptive_step,
            reparam_every=reparam,
        )
    refined.metadata["initializer_objective"] = coarse.objective
    refined.metadata["initializer_points"] = int(coarse.s.size)
    return refined


__all__ = [
    "MFEPPath",
    "GridMFEP",
    "NEBMFEP",
    "compute_mfep_profile_1d",
]
