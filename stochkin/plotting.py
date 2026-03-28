"""Plotting utilities for stochkin.

Every public function in this module applies the stochkin publication
style (:func:`~stochkin.style.set_publication_style`) before drawing,
so that all output figures are publication-ready by default.

The style mirrors the Matplotlib rcParams used in the FES_2D.ipynb
analysis notebook (Arial font, inward ticks, white background, 300 dpi,
single-column figure width).
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm, Normalize

from .style import (
    publication_style,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    CBAR_LABEL_SIZE,
    CBAR_TICK_SIZE,
    TITLE_SIZE,
)

# ── helper ────────────────────────────────────────────────────────────
def _apply_pub_axes(ax, xlabel=None, ylabel=None, title=None):
    """Apply consistent tick/label formatting to an Axes object."""
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE,
                   direction="in", length=5, width=0.8)
    ax.tick_params(axis="both", which="minor",
                   direction="in", length=3, width=0.8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, size=LABEL_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, size=LABEL_SIZE)
    if title is not None:
        ax.set_title(title, size=TITLE_SIZE)


def _apply_pub_cbar(cbar, label=None):
    """Format a colorbar to match the publication style."""
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    if label is not None:
        cbar.set_label(label, fontsize=CBAR_LABEL_SIZE)


# =====================================================================
# plot_results
# =====================================================================
def plot_results(times, positions, velocities, energies, bins=50):
    """Basic diagnostic plots for a 2D Langevin trajectory.

    Four panels: trajectory, energy vs time, position histogram,
    energy histogram.
    """
    with publication_style():
        # Trajectory
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        ax.plot(positions[:, 0], positions[:, 1], "-o", markersize=2)
        _apply_pub_axes(ax, "x₁", "x₂", "2D Langevin trajectory")
        fig.tight_layout()

        # Energy vs time
        fig2, ax2 = plt.subplots(figsize=(3.3, 2.8))
        ax2.plot(times, energies, "-")
        _apply_pub_axes(ax2, "time", "Total Energy", "Energy vs Time")
        fig2.tight_layout()

        # Position histogram
        fig3, ax3 = plt.subplots(figsize=(3.3, 2.8))
        h = ax3.hist2d(positions[:, 0], positions[:, 1], bins=bins, cmap="viridis")
        _apply_pub_cbar(fig3.colorbar(h[3], ax=ax3), label="counts")
        _apply_pub_axes(ax3, "x₁", "x₂", "Position distribution")
        fig3.tight_layout()

        # Energy histogram
        fig4, ax4 = plt.subplots(figsize=(3.3, 2.8))
        ax4.hist(energies, bins=bins, alpha=0.7)
        _apply_pub_axes(ax4, "Energy", "Frequency", "Energy distribution")
        fig4.tight_layout()

        plt.show()


# =====================================================================
# plot_mfpt_matrix
# =====================================================================
def plot_mfpt_matrix(
    mfpt_network_results,
    log10=False,
    cmap="magma",
    figsize=(3.3, 2.8),
    title="MFPT matrix τ(i→j)",
):
    """Plot MFPT(i→j) as a heatmap.

    Parameters
    ----------
    mfpt_network_results : dict
        Output of ``compute_mfpt_network``.
    log10 : bool
        If True, plot log₁₀(τᵢⱼ).
    cmap : str
        Colormap.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    """
    with publication_style():
        tau = np.array(mfpt_network_results["mfpt_matrix"], dtype=float)
        n = tau.shape[0]

        tau_masked = np.ma.masked_array(tau, mask=False)
        for i in range(n):
            tau_masked[i, i] = np.ma.masked

        if log10:
            with np.errstate(divide="ignore", invalid="ignore"):
                data = np.log10(tau_masked)
            clabel = r"$\log_{10} \tau_{ij}$"
        else:
            data = tau_masked
            clabel = r"$\tau_{ij}$"

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal")
        _apply_pub_cbar(fig.colorbar(im, ax=ax), label=clabel)
        _apply_pub_axes(ax, "j (target basin)", "i (start basin)", title)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))

        fig.tight_layout()
        plt.show()


# =====================================================================
# plot_fp_solution_vs_boltzmann
# =====================================================================
def plot_fp_solution_vs_boltzmann(
    fp_result,
    beta=1.0,
    log=True,
    cmap="viridis",
    figsize=None,
):
    """Compare FP steady-state with Boltzmann distribution.

    Parameters
    ----------
    fp_result : dict
        Output of ``solve_fp_steady_state``.
    beta : float
        Inverse temperature.
    log : bool
        If True, also show log-ratio panel.
    cmap : str
        Colormap.
    figsize : tuple, optional
        Figure size (default auto-sized for 2 or 3 panels).
    """
    with publication_style():
        xs = fp_result["xs"]
        ys = fp_result["ys"]
        p_grid = fp_result["p_grid"]
        U_grid = fp_result["U_grid"]

        X, Y = np.meshgrid(xs, ys, indexing="ij")

        boltz = np.exp(-beta * U_grid)
        boltz /= np.sum(boltz)

        ncols = 3 if log else 2
        if figsize is None:
            figsize = (3.3 * ncols, 2.8)

        fig, axes = plt.subplots(1, ncols, figsize=figsize)

        im0 = axes[0].contourf(X, Y, p_grid, levels=40, cmap=cmap)
        _apply_pub_cbar(fig.colorbar(im0, ax=axes[0]))
        _apply_pub_axes(axes[0], "x", "y", "FP steady p(x)")

        im1 = axes[1].contourf(X, Y, boltz, levels=40, cmap=cmap)
        _apply_pub_cbar(fig.colorbar(im1, ax=axes[1]))
        _apply_pub_axes(axes[1], "x", "y", r"Boltzmann $\propto e^{-\beta U}$")

        if log:
            with np.errstate(divide="ignore"):
                log_ratio = np.log(p_grid + 1e-30) - np.log(boltz + 1e-30)
            im2 = axes[2].contourf(X, Y, log_ratio, levels=40, cmap="coolwarm")
            _apply_pub_cbar(fig.colorbar(im2, ax=axes[2]))
            _apply_pub_axes(axes[2], "x", "y",
                            r"$\ln p_{\rm FP} - \ln p_{\rm Boltz}$")

        fig.tight_layout()
        plt.show()


# =====================================================================
# plot_basin_network
# =====================================================================
def plot_basin_network(
    basin_network,
    levels=40,
    fes_cmap="viridis",
    basin_cmap="tab20",
    alpha_basins=0.35,
    show_minima=True,
    annotate_ids=True,
    figsize=(3.3, 2.8),
):
    """Plot the FES with overlaid basin partition and minima.

    Parameters
    ----------
    basin_network : BasinNetwork
        As returned by ``detect_basins_for_mfpt``.
    levels : int
        Number of contour levels.
    fes_cmap, basin_cmap : str
        Colormaps.
    alpha_basins : float
        Basin overlay transparency.
    show_minima, annotate_ids : bool
        Show minimum markers / basin-id labels.
    figsize : tuple
        Figure size.
    """
    with publication_style():
        xs = basin_network.xs
        ys = basin_network.ys
        U = basin_network.U
        labels = basin_network.labels

        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = np.ma.masked_invalid(U)

        fig, ax = plt.subplots(figsize=figsize)

        cf = ax.contourf(X, Y, Z, levels=levels, cmap=fes_cmap)
        _apply_pub_cbar(fig.colorbar(cf, ax=ax), label="FES (kJ/mol)")

        label_mask = np.ma.masked_where(labels < 0, labels)
        n_basins = basin_network.n_basins
        bounds = np.arange(-1, n_basins + 1, 1)
        n_bins = len(bounds) - 1
        cmap_b = plt.cm.get_cmap("tab20", n_bins)
        norm = BoundaryNorm(bounds, ncolors=n_bins)

        ax.imshow(
            label_mask.T,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            cmap=cmap_b,
            norm=norm,
            alpha=alpha_basins,
            aspect="auto",
        )

        if show_minima:
            for b in basin_network.basins:
                ax.scatter(
                    b.minimum[0], b.minimum[1],
                    s=60, c="k", edgecolors="white", linewidths=1.0, zorder=5,
                )
                if annotate_ids:
                    ax.text(
                        b.minimum[0], b.minimum[1], f"{b.id}",
                        color="white", fontsize=9, ha="center", va="center",
                        zorder=6,
                    )

        _apply_pub_axes(ax, "x", "y", "Basins on FES")
        fig.tight_layout()
        plt.show()


# =====================================================================
# plot_central_well_barrier_ring
# =====================================================================
def plot_central_well_barrier_ring(
    a=1.0,
    b=1.0,
    A=0.5,
    sigma=0.5,
    r_max=2.0,
    n_points=400,
    grid_size=200,
):
    """Plot the radial profile + 2D landscape of the ring-barrier potential."""
    with publication_style():
        rs = np.linspace(0, r_max, n_points)
        U_radial = b * rs**4 - a * rs**2 - A * np.exp(-rs**2 / sigma**2)

        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        ax.plot(rs, U_radial)
        _apply_pub_axes(ax, "r", "U(r)", "Radial profile")
        fig.tight_layout()

        x = np.linspace(-r_max, r_max, grid_size)
        y = np.linspace(-r_max, r_max, grid_size)
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2
        U = b * R2**2 - a * R2 - A * np.exp(-R2 / sigma**2)

        fig2, ax2 = plt.subplots(figsize=(3.3, 2.8))
        cp = ax2.contourf(X, Y, U, levels=50, cmap="viridis")
        _apply_pub_cbar(fig2.colorbar(cp, ax=ax2), label="U(x)")
        _apply_pub_axes(ax2, "x₁", "x₂", "2D potential")
        ax2.set_aspect("equal")
        fig2.tight_layout()

        plt.show()


# =====================================================================
# plot_2d_fes  –  PLUMED 2D FES contour (like notebook plot_2d_contourf_MAX)
# =====================================================================
def plot_2d_fes(
    data_path,
    *,
    save_path=None,
    levels=10,
    fes_max=None,
    delta=90,
    reweight=False,
    invert=False,
    xlim=None,
    ylim=None,
    auto_zoom=True,
    zoom_pad=0.08,
    cmap_name="rainbow",
    swap_xy=True,
    show_cbar=True,
    cbar_label="FES (kJ/mol)",
    xlabel=None,
    ylabel=None,
    pathways=None,
    pathway_style=None,
    pathway_markers=False,
    pathway_every=1,
    pathway_labels=None,
    figsize=(4, 3),
    ax=None,
):
    """Plot a PLUMED-format 2D FES as a filled contour.

    This is a cleaned-up version of the ``plot_2d_contourf_MAX`` helper
    from the FES_2D.ipynb notebook, with publication styling applied
    automatically.

    Parameters
    ----------
    data_path : str or Path
        Path to a PLUMED ``sum_hills`` 2D FES file.
    save_path : str or Path, optional
        If given, save the figure to this path at 300 dpi.
    levels : int
        Number of contour levels.
    fes_max : float, optional
        Fixed colour-scale maximum (kJ/mol).  Also used as the masking
        threshold if *delta* is not explicitly set.
    delta : float
        Percentile threshold for masking (default 90).
    reweight : bool
        Subtract the minimum from the FES.
    swap_xy : bool
        Swap x/y axes (matches the notebook convention CN(Cl) vs CN(O)).
    pathways : list, optional
        MFEP overlay data (arrays or file paths).
    ax : matplotlib Axes, optional
        Draw onto an existing Axes instead of creating a new figure.

    Returns
    -------
    fig, ax : if *ax* was None
    ax       : if an existing *ax* was passed
    """
    with publication_style():
        data = np.genfromtxt(data_path, comments="#")
        data = data[np.all(np.isfinite(data[:, :3]), axis=1)]
        x, y, z = data[:, 0], data[:, 1], data[:, 2].copy()
        if reweight:
            z -= np.min(z)

        x_unique, y_unique = np.unique(x), np.unique(y)
        nx, ny = len(x_unique), len(y_unique)
        Z = z.reshape(ny, nx)

        if swap_xy:
            X, Y = np.meshgrid(y_unique, x_unique)
            Zplot, x_axis, y_axis = Z.T, y_unique, x_unique
            _xlabel = xlabel or "CN(Cl)"
            _ylabel = ylabel or "CN(O)"
        else:
            X, Y = np.meshgrid(x_unique, y_unique)
            Zplot, x_axis, y_axis = Z, x_unique, y_unique
            _xlabel = xlabel or "CV₁"
            _ylabel = ylabel or "CV₂"

        thr = float(fes_max) if fes_max is not None else float(np.percentile(Zplot, delta))
        Zm = np.ma.masked_where(Zplot >= thr, Zplot)

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("white")

        vmax = float(fes_max) if fes_max is not None else (
            float(np.ma.max(Zm)) if np.ma.count(Zm) > 0 else float(np.max(Zplot))
        )
        lev = np.linspace(0.0, vmax, int(levels))
        norm = Normalize(vmin=0.0, vmax=vmax)

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        cf = ax.contourf(X, Y, Zm, levels=lev, cmap=cmap, norm=norm, extend="max")

        # Pathways overlay
        if pathways is not None:
            if pathway_style is None:
                pathway_style = dict(lw=1.0, alpha=1.0)
            if not isinstance(pathways, (list, tuple)):
                pathways = [pathways]
            for k, p in enumerate(pathways):
                P = np.genfromtxt(p, comments="#") if isinstance(p, str) else np.asarray(p)
                px, py = P[:, 1], P[:, 2]
                xplot, yplot = (py, px) if swap_xy else (px, py)
                ax.plot(xplot, yplot, color="k", **pathway_style)
                if pathway_markers:
                    ax.plot(xplot[::pathway_every], yplot[::pathway_every],
                            ls="none", marker="o", ms=3, color="k",
                            alpha=pathway_style.get("alpha", 1.0))

        if show_cbar:
            cbar = fig.colorbar(cf, ax=ax)
            _apply_pub_cbar(cbar, label=cbar_label)

        # Auto zoom
        if auto_zoom and xlim is None and ylim is None and np.ma.count(Zm) > 0:
            jj, ii = np.where(~Zm.mask)
            xmin, xmax = x_axis[ii.min()], x_axis[ii.max()]
            ymin, ymax = y_axis[jj.min()], y_axis[jj.max()]
            dx = (xmax - xmin) or 1.0
            dy = (ymax - ymin) or 1.0
            ax.set_xlim(xmin - zoom_pad * dx, xmax + zoom_pad * dx)
            ax.set_ylim(ymin - zoom_pad * dy, ymax + zoom_pad * dy)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if invert:
            ax.invert_xaxis()

        _apply_pub_axes(ax, _xlabel, _ylabel)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300)

        if own_fig:
            return fig, ax
        return ax


# =====================================================================
# draw_barrier_arrows  –  ported from FES_2D.ipynb
# =====================================================================
def draw_barrier_arrows(
    ax,
    x,
    y_top,
    y0,
    *,
    y_bottom=None,
    label=True,
    label_fmt="{:.1f}",
    label_side="right",
    label_dx=0.05,
    label_dy=0.0,
    label_x_overrides=None,
    label_y_overrides=None,
    y_override_mode="abs",
    arrowprops=None,
    text_kwargs=None,
):
    """Draw double-headed barrier-height arrows on an Axes.

    Ported from the ``draw_barrier_arrows`` helper in FES_2D.ipynb.

    Parameters
    ----------
    ax : matplotlib Axes
    x : float or array
        Horizontal position(s) of the arrow(s).
    y_top : float or array
        Top of each arrow (barrier peak FES value).
    y0 : float
        Default baseline (bottom) for the arrows.
    y_bottom : float or array, optional
        Per-arrow bottom override.
    label : bool
        Annotate the barrier height ΔF.
    label_side : ``'right'`` or ``'left'``
        Side on which to place the label text.
    """
    x = np.atleast_1d(x)
    y_top = np.atleast_1d(y_top) if np.ndim(y_top) else np.full_like(x, y_top, dtype=float)
    y_bottom = (np.atleast_1d(y_bottom) if np.ndim(y_bottom) else np.full_like(x, y_bottom, dtype=float)) if y_bottom is not None else np.full_like(x, y0, dtype=float)

    base_ap = dict(arrowstyle="<->", ls="--", lw=0.75, color="0.2")
    if arrowprops:
        base_ap.update(arrowprops)

    side = label_side.lower()
    default_dx = abs(label_dx) if side == "right" else -abs(label_dx)
    default_ha = "left" if side == "right" else "right"
    base_tk = dict(ha=default_ha, va="center", fontsize=LEGEND_SIZE)
    if text_kwargs:
        base_tk.update(text_kwargs)

    label_x_overrides = label_x_overrides or {}
    label_y_overrides = label_y_overrides or {}

    out = []
    for i, (xi, yb, yt) in enumerate(zip(x, y_bottom, y_top)):
        ann = ax.annotate("", xy=(xi, yb), xytext=(xi, yt), arrowprops=base_ap)
        txt = None
        if label:
            dF = yt - yb
            mid_y = 0.5 * (yb + yt) + label_dy
            dx = label_x_overrides.get(i, default_dx)
            if i in label_y_overrides:
                mid_y = mid_y * label_y_overrides[i] if y_override_mode == "mul" else label_y_overrides[i]
            txt = ax.text(xi + dx, mid_y, label_fmt.format(dF), **base_tk)
        out.append((ann, txt))
    return out
