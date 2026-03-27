
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm



def plot_results(times, positions, velocities, energies, bins=50):
    """
    Basic plotting of a 2D trajectory, energy vs time,
    position histogram, and energy histogram.
    """
    # Trajectory in 2D
    plt.figure()
    plt.plot(positions[:, 0], positions[:, 1], "-o", markersize=2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("2D Langevin trajectory")

    # Energy vs time
    plt.figure()
    plt.plot(times, energies, "-")
    plt.xlabel("time")
    plt.ylabel("Total Energy")
    plt.title("Energy vs Time")

    # Histogram of positions
    plt.figure()
    plt.hist2d(positions[:, 0], positions[:, 1], bins=bins, cmap="viridis")
    plt.colorbar(label="counts")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Position distribution")

    # Histogram of energies
    plt.figure()
    plt.hist(energies, bins=bins, alpha=0.7)
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.title("Energy distribution")
    plt.show()

def plot_mfpt_matrix(
    mfpt_network_results,
    log10=False,
    cmap="magma",
    figsize=(5, 4),
    title="MFPT matrix τ(i→j)",
):
    """
    Plot MFPT(i->j) from compute_mfpt_network as a heatmap.

    Parameters
    ----------
    mfpt_network_results : dict
        Output of compute_mfpt_network.
    log10 : bool
        If True, plot log10(τ_ij) instead of τ_ij.
    cmap : str
        Colormap for the matrix.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    """
    tau = np.array(mfpt_network_results["mfpt_matrix"], dtype=float)
    n = tau.shape[0]

    # mask self-transitions (i->i) to avoid clutter
    tau_masked = np.ma.masked_array(tau, mask=False)
    for i in range(n):
        tau_masked[i, i] = np.ma.masked

    if log10:
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.log10(tau_masked)
        label = r"$\log_{10} \tau_{ij}$"
    else:
        data = tau_masked
        label = r"$\tau_{ij}$"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)

    ax.set_xlabel("j (target basin)")
    ax.set_ylabel("i (start basin)")
    ax.set_title(title)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    plt.tight_layout()
    plt.show()

def plot_fp_solution_vs_boltzmann(
    fp_result,
    beta=1.0,
    log=True,
    cmap="viridis",
    figsize=(14, 4),
):
    """
    Compare Fokker–Planck steady solution with Boltzmann distribution.

    Parameters
    ----------
    fp_result : dict
        Output of solve_fp_steady_state.
    beta : float
        Inverse temperature used in the FP solver.
    log : bool
        If True, also plot log p.
    """
    xs = fp_result["xs"]
    ys = fp_result["ys"]
    p_grid = fp_result["p_grid"]
    U_grid = fp_result["U_grid"]

    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # Boltzmann distribution (unnormalized)
    boltz = np.exp(-beta * U_grid)
    boltz /= np.sum(boltz)  # normalize on grid (up to cell volumes factor)

    fig, axes = plt.subplots(1, 3 if log else 2, figsize=figsize)

    im0 = axes[0].contourf(X, Y, p_grid, levels=40, cmap=cmap)
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("FP steady p(x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im1 = axes[1].contourf(X, Y, boltz, levels=40, cmap=cmap)
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Boltzmann ∝ exp(-βU)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    if log:
        with np.errstate(divide="ignore"):
            log_ratio = np.log(p_grid + 1e-30) - np.log(boltz + 1e-30)

        im2 = axes[2].contourf(X, Y, log_ratio, levels=40, cmap="coolwarm")
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title("log p_FP - log p_Boltz")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")

    plt.tight_layout()
    plt.show()


def plot_basin_network(
    basin_network,
    levels=40,
    fes_cmap="viridis",
    basin_cmap="tab20",
    alpha_basins=0.35,
    show_minima=True,
    annotate_ids=True,
    figsize=(6, 5),
):
    """
    Plot the FES and overlay the basin partition + minima positions.

    Parameters
    ----------
    basin_network : BasinNetwork
        As returned by detect_basins_for_mfpt.
    levels : int
        Number of contour levels for FES.
    fes_cmap : str
        Colormap for FES.
    basin_cmap : str
        Colormap for basin labels (categorical).
    alpha_basins : float
        Transparency of basin colors on top of FES.
    show_minima : bool
        Scatter local minima.
    annotate_ids : bool
        Write basin ids near minima.
    figsize : tuple
        Matplotlib figure size.
    """
    xs = basin_network.xs
    ys = basin_network.ys
    U = basin_network.U
    labels = basin_network.labels

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.ma.masked_invalid(U)

    fig, ax = plt.subplots(figsize=figsize)

    # FES background
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=fes_cmap)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("U(x) / FES")

    # Basin labels (mask out -1)
    label_mask = np.ma.masked_where(labels < 0, labels)
    n_basins = basin_network.n_basins

    # Discrete bounds: one bin per basin plus a "background" bin
    bounds = np.arange(-1, n_basins + 1, 1)  # [-1, 0, 1, ..., n_basins]
    n_bins = len(bounds) - 1                 # = n_basins + 1

    # Build a colormap with at least as many colors as bins
    cmap_b = plt.cm.get_cmap("tab20", n_bins)

    # BoundaryNorm expects ncolors >= number of bins
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

    # Minima
    if show_minima:
        for b in basin_network.basins:
            ax.scatter(
                b.minimum[0],
                b.minimum[1],
                s=60,
                c="k",
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
            )
            if annotate_ids:
                ax.text(
                    b.minimum[0],
                    b.minimum[1],
                    f"{b.id}",
                    color="white",
                    fontsize=9,
                    ha="center",
                    va="center",
                    zorder=6,
                )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Basins on FES")
    plt.tight_layout()
    plt.show()


def plot_central_well_barrier_ring(
    a=1.0,
    b=1.0,
    A=0.5,
    sigma=0.5,
    r_max=2.0,
    n_points=400,
    grid_size=200,
):
    """
    Plot the radial profile and 2D landscape of the
    central_well_barrier_ring_potential.
    """
    # Radial profile
    rs = np.linspace(0, r_max, n_points)
    U_radial = b * rs ** 4 - a * rs ** 2 - A * np.exp(-rs ** 2 / sigma ** 2)

    plt.figure(figsize=(6, 4))
    plt.plot(rs, U_radial)
    plt.xlabel("r")
    plt.ylabel("U(r)")
    plt.title("Radial profile of potential")
    plt.grid(True)

    # 2D potential landscape
    x = np.linspace(-r_max, r_max, grid_size)
    y = np.linspace(-r_max, r_max, grid_size)
    X, Y = np.meshgrid(x, y)
    R2 = X ** 2 + Y ** 2
    U = b * R2 ** 2 - a * R2 - A * np.exp(-R2 / sigma ** 2)

    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, U, levels=50, cmap="viridis")
    plt.colorbar(cp, label="U(x)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("2D potential landscape")
    plt.axis("equal")
    plt.show()
