#!/usr/bin/env python3
"""Example 05 – Pairwise MFEP paths and CTMC rates between all basins
======================================================================

This script demonstrates the full **multi-basin MFEP–CTMC pipeline** on
a 2D free-energy surface, with emphasis on **visualising every pairwise
path** and the corresponding 1D free-energy profile F(s).

Workflow
--------
1. Load a 2D PLUMED FES and detect all basins.
2. For every pair of basins, compute the minimum free-energy path (MFEP).
3. Show each MFEP overlaid on the 2D FES and its 1D arc-length profile.
4. Assemble the full N×N CTMC rate matrix and display exit times.

Generated figures
-----------------
    05_all_paths_on_fes.png    – all MFEPs overlaid on the 2D FES
    05_pairwise_profiles.png   – 1D F(s) profile for each pair of basins
    05_rate_matrix.png         – full rate-matrix heatmap
    05_exit_times.png          – bar chart of mean exit times

Usage
-----
    # With the bundled synthetic 3-basin surface:
    python 05_pairwise_mfep_paths.py  data/synthetic_2d_fes.dat

    # Custom FES and diffusion coefficient:
    python 05_pairwise_mfep_paths.py  my_fes_2d.dat --D-s 0.1 -T 300

    # Regenerate the synthetic data first (if needed):
    python generate_synthetic_data.py
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import stochkin as sk
from stochkin.mfep import compute_mfep_profile_1d
from stochkin.style import publication_style, LABEL_SIZE
from stochkin.plotting import _apply_pub_axes, _apply_pub_cbar


# ── CLI ──────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser(
        description="2D FES → pairwise MFEPs → 1D profiles → CTMC rates")
    p.add_argument("fes2d", help="Path to a 2D PLUMED FES file")
    p.add_argument("--D-s", type=float, default=0.04,
                   help="Diffusion along arc-length  [arc-length² ps⁻¹]")
    p.add_argument("-T", type=float, default=300.0,
                   help="Temperature in K")
    p.add_argument("--neb-images", type=int, default=120)
    p.add_argument("--neb-steps", type=int, default=8000)
    p.add_argument("--max-basins", type=int, default=None,
                   help="Keep only the N deepest basins  (default: all)")
    p.add_argument("--core-fraction", type=float, default=0.05)
    p.add_argument("--out-prefix", default="05")
    return p.parse_args()


# ── Colour helpers ───────────────────────────────────────────────────
_PAIR_COLOURS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabebe",  # pink
    "#469990",  # teal
]


def _colour_for(idx: int) -> str:
    return _PAIR_COLOURS[idx % len(_PAIR_COLOURS)]


# ── Main ─────────────────────────────────────────────────────────────
def main():
    args = _parse()
    prefix = args.out_prefix

    # ── 1. Run the multi-MFEP CTMC workflow ──────────────────────────
    print(f"Loading 2D FES: {args.fes2d}")
    result = sk.run_multi_mfep_ctmc(
        fes2d_path=args.fes2d,
        D_s=args.D_s,
        T=args.T,
        neb_images=args.neb_images,
        neb_steps=args.neb_steps,
        max_basins=args.max_basins,
        core_fraction=args.core_fraction,
    )

    basin_net = result["basin_network"]
    basins    = basin_net.basins
    n         = len(basins)
    bids      = result["basin_ids"]
    legs      = result["legs"]
    kT        = result["kT"]

    K_ns   = result["K_ps"] * 1000        # ps⁻¹ → ns⁻¹
    tau_ns = result["exit_ps"] / 1000      # ps  → ns

    print(f"\n{'='*60}")
    print(f"Detected {n} basins")
    for b in basins:
        print(f"  Basin {b.id}: minimum ({b.minimum[0]:.2f}, "
              f"{b.minimum[1]:.2f}),  F_min = {b.f_min:.2f} kJ/mol")
    print(f"\nRate matrix K [ns⁻¹]:\n{K_ns}")
    print(f"Mean exit times [ns]: {tau_ns}")

    # ── 2. Reload the raw grid for plotting ──────────────────────────
    x_grid, y_grid, fes_grid = sk.load_plumed_fes_2d(
        args.fes2d, verbose=False)

    # Pair labels and their legs
    pair_keys = sorted(legs.keys())  # (i, j) tuples
    n_pairs = len(pair_keys)

    # ── 3. Figure 1: all paths on the 2D FES ────────────────────────
    with publication_style():
        fig, ax = plt.subplots(figsize=(4.5, 3.6))

        cs = ax.contourf(
            x_grid, y_grid, (fes_grid / kT).T,
            levels=np.linspace(0, 15, 30), cmap="rainbow_r",
        )
        cbar = fig.colorbar(cs, ax=ax)
        _apply_pub_cbar(cbar, label=r"$F\,/\,k_\mathrm{B}T$")

        # Basin minima
        for b in basins:
            ax.plot(*b.minimum, "o", color="white", ms=10, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.6)
            ax.text(b.minimum[0] + 0.15, b.minimum[1] + 0.15,
                    f"B{b.id}", color="white", fontsize=10,
                    fontweight="bold", zorder=6,
                    path_effects=[
                        pe.withStroke(
                            linewidth=2, foreground="black")])

        # MFEP legs
        for k, (i, j) in enumerate(pair_keys):
            path = legs[(i, j)]["mfep_path"]
            col = _colour_for(k)
            ax.plot(path.x, path.y, "-", color=col, lw=2.0, alpha=0.9,
                    label=f"B{basins[i].id}→B{basins[j].id}", zorder=4)

        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        _apply_pub_axes(ax, r"CV$_1$", r"CV$_2$",
                        "Pairwise MFEPs on 2D FES")
        fig.tight_layout()
        fig.savefig(f"{prefix}_all_paths_on_fes.png", dpi=300)
    print(f"\nSaved {prefix}_all_paths_on_fes.png")

    # ── 4. Figure 2: 1D F(s) profiles for each pair ─────────────────
    n_cols = min(n_pairs, 3)
    n_rows = int(np.ceil(n_pairs / n_cols))

    with publication_style():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.3 * n_cols, 2.6 * n_rows),
            squeeze=False,
        )

        for k, (i, j) in enumerate(pair_keys):
            row, col = divmod(k, n_cols)
            ax = axes[row, col]
            leg = legs[(i, j)]
            path = leg["mfep_path"]

            s  = path.s
            F  = path.F - np.nanmin(path.F)  # shift to zero

            colour = _colour_for(k)
            ax.fill_between(s, F / kT, alpha=0.20, color=colour)
            ax.plot(s, F / kT, "-", color=colour, lw=1.5)

            # Mark detected basins on the 1D profile (if available)
            bn_1d = leg.get("basin_network")
            if bn_1d is not None:
                for b1d in bn_1d.basins:
                    idx_min = np.argmin(np.abs(s - b1d.minimum))
                    ax.axvline(s[idx_min], ls=":", color="grey",
                               lw=0.7, alpha=0.6)

            # Rate annotation
            K_leg = leg["K"]
            if K_leg.shape[0] >= 2:
                k_fwd = K_leg[0, -1]   # forward rate
                k_bwd = K_leg[-1, 0]   # backward rate
                ax.text(0.97, 0.95,
                        f"$k_{{\\rightarrow}}={k_fwd:.2e}$\n"
                        f"$k_{{\\leftarrow}}={k_bwd:.2e}$",
                        transform=ax.transAxes, fontsize=6,
                        va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", alpha=0.8))

            _apply_pub_axes(
                ax,
                xlabel="Arc-length  $s$",
                ylabel=r"$F(s)\,/\,k_\mathrm{B}T$",
                title=f"B{basins[i].id} → B{basins[j].id}",
            )

        # Hide unused subplots
        for k in range(n_pairs, n_rows * n_cols):
            row, col = divmod(k, n_cols)
            axes[row, col].set_visible(False)

        fig.tight_layout()
        fig.savefig(f"{prefix}_pairwise_profiles.png", dpi=300)
    print(f"Saved {prefix}_pairwise_profiles.png")

    # ── 5. Figure 3: rate-matrix heatmap ─────────────────────────────
    K_abs = np.abs(K_ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        Klog = np.where(K_abs > 0, np.log10(K_abs), np.nan)

    with publication_style():
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        im = ax.imshow(Klog, cmap="magma_r", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"B{i}" for i in bids])
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"B{i}" for i in bids])

        # Annotate cells with numeric values
        for ii in range(n):
            for jj in range(n):
                if ii != jj and K_ns[ii, jj] > 0:
                    ax.text(jj, ii, f"{K_ns[ii, jj]:.1e}",
                            ha="center", va="center", fontsize=5,
                            color="white" if Klog[ii, jj] < np.nanmedian(
                                Klog[np.isfinite(Klog)]) else "black")

        cbar = fig.colorbar(im, ax=ax)
        _apply_pub_cbar(cbar, label=r"$\log_{10}|K_{ij}|$  [ns$^{-1}$]")
        _apply_pub_axes(ax, title=f"Rate matrix  ({n} basins)")
        fig.tight_layout()
        fig.savefig(f"{prefix}_rate_matrix.png", dpi=300)
    print(f"Saved {prefix}_rate_matrix.png")

    # ── 6. Figure 4: mean exit times ─────────────────────────────────
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        bars = ax.bar(range(n), tau_ns, color=[_colour_for(k) for k in range(n)],
                      edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"B{i}" for i in bids])
        for bar, t in zip(bars, tau_ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{t:.1f}", ha="center", va="bottom", fontsize=8)
        _apply_pub_axes(ax, xlabel="Basin", ylabel="Exit time  [ns]",
                        title="Mean exit times")
        fig.tight_layout()
        fig.savefig(f"{prefix}_exit_times.png", dpi=300)
    print(f"Saved {prefix}_exit_times.png")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Pairwise summary")
    print(f"{'Pair':<12} {'k_fwd':>12} {'k_bwd':>12}  {'barrier_fwd':>12}")
    print("-" * 60)
    for i, j in pair_keys:
        leg = legs[(i, j)]
        K_leg = leg["K"]
        path = leg["mfep_path"]
        F = path.F - np.nanmin(path.F)
        barrier = np.nanmax(F)
        if K_leg.shape[0] >= 2:
            k_fwd = K_leg[0, -1]
            k_bwd = K_leg[-1, 0]
        else:
            k_fwd = k_bwd = float("nan")
        print(f"B{basins[i].id}→B{basins[j].id}"
              f"      {k_fwd:12.4e} {k_bwd:12.4e}  {barrier:10.2f} kJ/mol")
    print("=" * 60)


if __name__ == "__main__":
    main()
