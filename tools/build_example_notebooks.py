#!/usr/bin/env python3
"""Generate Jupyter notebooks that mirror the Python example scripts.

The notebooks are intentionally built from small templates rather than
hand-edited JSON so they can be regenerated whenever the example scripts
or default parameters change.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def _source(text: str) -> list[str]:
    body = textwrap.dedent(text).lstrip("\n").rstrip() + "\n"
    return body.splitlines(keepends=True)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = """
from pathlib import Path
import sys


def find_repo_root(start=None):
    current = (start or Path.cwd()).resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "stochkin").is_dir():
            return candidate
    raise RuntimeError("Could not locate the repository root from the current working directory.")


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "examples" / "data"
OUT_DIR = ROOT / "notebooks" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Repository root: {ROOT}")
print(f"Notebook output directory: {OUT_DIR}")
"""


NOTEBOOK_SPECS: dict[str, list[dict]] = {
    "00_generate_synthetic_data.ipynb": [
        md(
            """
            # Example 00 - Generate Synthetic Data

            This notebook regenerates the bundled synthetic datasets used by the
            other examples. It wraps `examples/generate_synthetic_data.py` so the
            same reference files can be recreated from Jupyter.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            import subprocess

            cmd = [
                sys.executable,
                str(ROOT / "examples" / "generate_synthetic_data.py"),
                "--plot",
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            """
        ),
        code(
            """
            generated = sorted(path.relative_to(ROOT).as_posix() for path in DATA_DIR.iterdir())
            generated
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            figure_names = ["diagnostic_1d.png", "diagnostic_2d.png"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            for ax, name in zip(axes, figure_names):
                img_path = DATA_DIR / name
                ax.imshow(mpimg.imread(img_path))
                ax.set_title(name)
                ax.axis("off")

            fig.tight_layout()
            plt.show()
            """
        ),
    ],
    "01_analytic_doublewell.ipynb": [
        md(
            """
            # Example 01 - Analytic Double-Well Potential

            This notebook mirrors `examples/01_analytic_doublewell.py`:

            1. Define a simple analytic 2D double-well potential.
            2. Detect basins automatically on the potential grid.
            3. Estimate bidirectional MFPTs from overdamped Langevin replicas.
            4. Plot the potential landscape and detected basin minima.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            import numpy as np
            import matplotlib.pyplot as plt
            import stochkin as sk

            from stochkin.plotting import _apply_pub_axes, _apply_pub_cbar
            from stochkin.style import publication_style
            """
        ),
        code(
            """
            KT = 0.5
            GAMMA = 1.0
            DT = 1e-3
            MAX_TIME = 5e4
            N_TRIALS = 200


            def double_well(x):
                x = np.asarray(x, dtype=float)
                U = (x[0] ** 2 - 1.0) ** 2 + x[1] ** 2
                F = np.array(
                    [
                        -4.0 * x[0] * (x[0] ** 2 - 1.0),
                        -2.0 * x[1],
                    ]
                )
                return U, F
            """
        ),
        code(
            """
            basin_net = sk.build_basin_network_from_potential(
                double_well,
                xlim=(-2.0, 2.0),
                ylim=(-1.5, 1.5),
                nx=101,
                ny=101,
            )

            print(f"Found {basin_net.n_basins} basins")
            for basin in basin_net.basins:
                print(f"Basin {basin.id}: minimum ~ {basin.minimum}")
            """
        ),
        code(
            """
            basin_a, basin_b = basin_net.basins[0], basin_net.basins[1]

            result = sk.compute_bidirectional_mfpt(
                double_well,
                lambda x: basin_net.which_basin(x) == basin_a.id,
                lambda x: basin_net.which_basin(x) == basin_b.id,
                kT=KT,
                gamma=GAMMA,
                n_trials=N_TRIALS,
                max_time=MAX_TIME,
                dt=DT,
                regime="overdamped",
                boundsA=basin_a.bounds,
                boundsB=basin_b.bounds,
                processes=1,
            )

            mfpt_01 = result["A_to_B"]["mean"]
            mfpt_10 = result["B_to_A"]["mean"]
            print(f"MFPT 0->1 = {mfpt_01:.2e} tau")
            print(f"MFPT 1->0 = {mfpt_10:.2e} tau")
            """
        ),
        code(
            """
            X = np.linspace(-2.0, 2.0, 200)
            Y = np.linspace(-1.5, 1.5, 200)
            Xg, Yg = np.meshgrid(X, Y, indexing="ij")
            U_grid = np.vectorize(lambda xi, yi: double_well([xi, yi])[0])(Xg, Yg)

            output_path = OUT_DIR / "01_doublewell.png"

            with publication_style():
                fig, ax = plt.subplots(figsize=(3.3, 2.8))
                cs = ax.contourf(
                    X,
                    Y,
                    U_grid.T / KT,
                    levels=np.linspace(0, 10, 20),
                    cmap="viridis_r",
                )
                cbar = fig.colorbar(cs, ax=ax)
                _apply_pub_cbar(cbar, label=r"$U / k_\\mathrm{B}T$")

                for basin in basin_net.basins:
                    mx, my = float(basin.minimum[0]), float(basin.minimum[1])
                    ax.plot(mx, my, "o", color="white", ms=8)
                    ax.text(mx, my, f" B{basin.id}", color="white", va="center", fontsize=9)

                _apply_pub_axes(ax, r"$x_1$", r"$x_2$", "Double-well potential")
                fig.tight_layout()
                fig.savefig(output_path, dpi=300)

            print(f"Saved {output_path.relative_to(ROOT)}")
            plt.show()
            """
        ),
    ],
    "02_1d_plumed_fes_ctmc.ipynb": [
        md(
            """
            # Example 02 - 1D PLUMED FES + Constant D

            This notebook mirrors `examples/02_1d_plumed_fes_ctmc.py` using the
            bundled synthetic 1D FES. It runs the high-level 1D CTMC workflow,
            inspects the rate matrix, and plots the basin partitioning.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import stochkin as sk

            from stochkin.plotting import _apply_pub_axes, _apply_pub_cbar
            from stochkin.style import LEGEND_SIZE, publication_style
            """
        ),
        code(
            """
            fes_path = DATA_DIR / "synthetic_1d_fes.dat"
            D = 0.05
            crop = (0.5, 9.5)
            T = 300.0
            resample_n = 500
            core_fraction = 0.05

            result = sk.run_1d_ctmc_from_plumed(
                fes_path=fes_path,
                D=D,
                T=T,
                crop=crop,
                resample_n=resample_n,
                core_fraction=core_fraction,
            )

            print(f"Basins: {result['basin_ids']}")
            """
        ),
        code(
            """
            basin_labels = [f"B{bid}" for bid in result["basin_ids"]]
            K_ns = pd.DataFrame(
                result["K_ps"] * 1000.0,
                index=basin_labels,
                columns=basin_labels,
            )
            exit_times_ns = pd.Series(
                result["exit_ps"] / 1000.0,
                index=basin_labels,
                name="exit_time_ns",
            )

            K_ns
            """
        ),
        code(
            """
            exit_times_ns
            """
        ),
        code(
            """
            s = result["s"]
            F = result["F"]
            labels = result["labels_full"]
            output_path = OUT_DIR / "02_ctmc_rates.png"

            with publication_style():
                cmap = plt.get_cmap("tab10")
                fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))

                ax = axes[0]
                ax.plot(s, F, "k-", lw=1.5)
                for bid in result["basin_ids"]:
                    mask = labels == bid
                    ax.fill_between(
                        s,
                        F,
                        where=mask,
                        alpha=0.25,
                        color=cmap(int(bid)),
                        label=f"B{bid}",
                    )
                _apply_pub_axes(ax, "CV", r"F  [kJ mol$^{-1}$]", "FES + basins")
                ax.legend(fontsize=LEGEND_SIZE, frameon=False)

                ax = axes[1]
                K_abs = np.abs(K_ns.to_numpy())
                with np.errstate(divide="ignore", invalid="ignore"):
                    Klog = np.where(K_abs > 0, np.log10(K_abs), np.nan)
                im = ax.imshow(Klog, cmap="magma_r", aspect="auto")
                ax.set_xticks(range(len(basin_labels)))
                ax.set_xticklabels(basin_labels)
                ax.set_yticks(range(len(basin_labels)))
                ax.set_yticklabels(basin_labels)
                cbar = fig.colorbar(im, ax=ax)
                _apply_pub_cbar(cbar, label=r"$\\log_{10}|K_{ij}|$  [ns$^{-1}$]")
                _apply_pub_axes(ax, title="Rate matrix")

                fig.tight_layout()
                fig.savefig(output_path, dpi=300)

            print(f"Saved {output_path.relative_to(ROOT)}")
            plt.show()
            """
        ),
    ],
    "03_1d_hummer_D_ctmc.ipynb": [
        md(
            """
            # Example 03 - 1D PLUMED FES + Hummer D(s)

            This notebook mirrors `examples/03_1d_hummer_D_ctmc.py` using the
            bundled synthetic FES and diffusion-profile CSV. It feeds a
            position-dependent diffusion profile into the 1D CTMC workflow.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import stochkin as sk

            from stochkin.plotting import _apply_pub_axes, _apply_pub_cbar
            from stochkin.style import publication_style
            """
        ),
        code(
            """
            fes_path = DATA_DIR / "synthetic_1d_fes.dat"
            d_csv = DATA_DIR / "synthetic_diffusion_profile.csv"
            crop = (0.5, 9.5)
            T = 300.0
            resample_n = 500
            core_fraction = 0.05

            result = sk.run_1d_ctmc_with_hummer_D(
                fes_path=fes_path,
                d_csv=d_csv,
                T=T,
                d_xcol="x_interface",
                d_col="D_med",
                d_grid="interface",
                crop=crop,
                resample_n=resample_n,
                core_fraction=core_fraction,
            )
            """
        ),
        code(
            """
            basin_labels = [f"B{bid}" for bid in result["basin_ids"]]
            K_ns = pd.DataFrame(
                result["K_ps"] * 1000.0,
                index=basin_labels,
                columns=basin_labels,
            )
            exit_times_ns = pd.Series(
                result["exit_ps"] / 1000.0,
                index=basin_labels,
                name="exit_time_ns",
            )

            K_ns
            """
        ),
        code(
            """
            exit_times_ns
            """
        ),
        code(
            """
            s = result["s"]
            F = result["F"]
            D_used = np.asarray(result["D_used"])

            out_fes = OUT_DIR / "03_fes_and_D.png"
            out_rates = OUT_DIR / "03_ctmc_rates.png"

            with publication_style():
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 4.2), sharex=True)
                ax1.plot(s, F, "k-", lw=1.5)
                _apply_pub_axes(ax1, ylabel=r"F  [kJ mol$^{-1}$]", title="Free-energy profile")

                ax2.plot(s, D_used, "b-", lw=1.5)
                _apply_pub_axes(
                    ax2,
                    xlabel="CV",
                    ylabel=r"$D(s)$  [CV$^2$ ps$^{-1}$]",
                    title="Hummer D(s) (interpolated)",
                )

                fig.tight_layout()
                fig.savefig(out_fes, dpi=300)

            with publication_style():
                K_abs = np.abs(K_ns.to_numpy())
                with np.errstate(divide="ignore", invalid="ignore"):
                    Klog = np.where(K_abs > 0, np.log10(K_abs), np.nan)

                fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
                im = axes[0].imshow(Klog, cmap="magma_r", aspect="auto")
                axes[0].set_xticks(range(len(basin_labels)))
                axes[0].set_xticklabels(basin_labels)
                axes[0].set_yticks(range(len(basin_labels)))
                axes[0].set_yticklabels(basin_labels)
                cbar = fig.colorbar(im, ax=axes[0])
                _apply_pub_cbar(cbar, label=r"$\\log_{10}|K_{ij}|$  [ns$^{-1}$]")
                _apply_pub_axes(axes[0], title="Rate matrix")

                axes[1].bar(range(len(basin_labels)), exit_times_ns.to_numpy(), color="steelblue")
                axes[1].set_yscale("log")
                axes[1].set_xticks(range(len(basin_labels)))
                axes[1].set_xticklabels(basin_labels)
                _apply_pub_axes(axes[1], ylabel="Exit time  [ns]", title="Mean exit times")

                fig.tight_layout()
                fig.savefig(out_rates, dpi=300)

            print(f"Saved {out_fes.relative_to(ROOT)}")
            print(f"Saved {out_rates.relative_to(ROOT)}")
            plt.show()
            """
        ),
    ],
    "04_mfep_ctmc.ipynb": [
        md(
            """
            # Example 04 - 2D FES to Multi-Basin MFEP CTMC

            This notebook mirrors `examples/04_mfep_ctmc.py` using the bundled
            synthetic 2D FES. It detects all basins, computes pairwise MFEP legs,
            assembles the CTMC rate matrix, and visualizes the full network.

            This is one of the heavier examples because each basin pair requires
            an MFEP and 1D arc-length reduction.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import stochkin as sk

            from stochkin.plotting import _apply_pub_axes, _apply_pub_cbar
            from stochkin.style import publication_style
            """
        ),
        code(
            """
            fes2d_path = DATA_DIR / "synthetic_2d_fes.dat"
            D_s = 0.05
            T = 300.0
            neb_images = 120
            neb_steps = 8000
            max_basins = None
            core_fraction = 0.05

            result = sk.run_multi_mfep_ctmc(
                fes2d_path=fes2d_path,
                D_s=D_s,
                T=T,
                neb_images=neb_images,
                neb_steps=neb_steps,
                max_basins=max_basins,
                core_fraction=core_fraction,
            )
            """
        ),
        code(
            """
            basin_labels = [f"B{bid}" for bid in result["basin_ids"]]
            K_ns = pd.DataFrame(
                result["K_ps"] * 1000.0,
                index=basin_labels,
                columns=basin_labels,
            )
            exit_times_ns = pd.Series(
                result["exit_ps"] / 1000.0,
                index=basin_labels,
                name="exit_time_ns",
            )

            print(f"Detected {len(basin_labels)} basins")
            for basin in result["basin_network"].basins:
                print(f"Basin {basin.id}: minimum {basin.minimum}, F_min = {basin.f_min:.2f} kJ/mol")
            """
        ),
        code(
            """
            K_ns
            """
        ),
        code(
            """
            exit_times_ns
            """
        ),
        code(
            """
            x_grid, y_grid, fes_grid = sk.load_plumed_fes_2d(fes2d_path, verbose=False)
            basin_net = result["basin_network"]
            kT = result["kT"]

            out_fes = OUT_DIR / "04_basins_on_fes.png"
            out_rates = OUT_DIR / "04_rate_matrix.png"
            out_exit = OUT_DIR / "04_exit_times.png"

            with publication_style():
                fig, ax = plt.subplots(figsize=(3.3, 2.8))
                cs = ax.contourf(
                    x_grid,
                    y_grid,
                    (fes_grid / kT).T,
                    levels=np.linspace(0, 15, 30),
                    cmap="rainbow_r",
                )
                cbar = fig.colorbar(cs, ax=ax)
                _apply_pub_cbar(cbar, label=r"$F / k_\\mathrm{B}T$")

                for basin in basin_net.basins:
                    ax.plot(*basin.minimum, "o", color="white", ms=10, zorder=5)
                    ax.text(
                        basin.minimum[0],
                        basin.minimum[1],
                        f" B{basin.id}",
                        color="white",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        zorder=6,
                    )

                for (_, _), leg in result["legs"].items():
                    path = leg["mfep_path"]
                    ax.plot(path.x, path.y, "--", color="white", lw=1.5, alpha=0.8)

                _apply_pub_axes(ax, r"CV$_1$", r"CV$_2$", "Basins + MFEPs on 2D FES")
                fig.tight_layout()
                fig.savefig(out_fes, dpi=300)

            with publication_style():
                K_abs = np.abs(K_ns.to_numpy())
                with np.errstate(divide="ignore", invalid="ignore"):
                    Klog = np.where(K_abs > 0, np.log10(K_abs), np.nan)

                fig, ax = plt.subplots(figsize=(3.3, 2.8))
                im = ax.imshow(Klog, cmap="magma_r", aspect="auto")
                ax.set_xticks(range(len(basin_labels)))
                ax.set_xticklabels(basin_labels)
                ax.set_yticks(range(len(basin_labels)))
                ax.set_yticklabels(basin_labels)
                cbar = fig.colorbar(im, ax=ax)
                _apply_pub_cbar(cbar, label=r"$\\log_{10}|K_{ij}|$  [ns$^{-1}$]")
                _apply_pub_axes(ax, title=f"Rate matrix ({len(basin_labels)} basins)")
                fig.tight_layout()
                fig.savefig(out_rates, dpi=300)

            with publication_style():
                fig, ax = plt.subplots(figsize=(3.3, 2.8))
                bars = ax.bar(range(len(basin_labels)), exit_times_ns.to_numpy(), color="steelblue")
                ax.set_yscale("log")
                ax.set_xticks(range(len(basin_labels)))
                ax.set_xticklabels(basin_labels)
                for bar, value in zip(bars, exit_times_ns.to_numpy()):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                _apply_pub_axes(ax, xlabel="Basin", ylabel="Exit time  [ns]", title="Mean exit times")
                fig.tight_layout()
                fig.savefig(out_exit, dpi=300)

            print(f"Saved {out_fes.relative_to(ROOT)}")
            print(f"Saved {out_rates.relative_to(ROOT)}")
            print(f"Saved {out_exit.relative_to(ROOT)}")
            plt.show()
            """
        ),
    ],
    "05_pairwise_mfep_paths.ipynb": [
        md(
            """
            # Example 05 - Pairwise MFEP Paths and CTMC Rates

            This notebook mirrors `examples/05_pairwise_mfep_paths.py`. It is
            the most detailed 2D example in the repository:

            1. Detect all basins on the synthetic 2D FES.
            2. Compute the MFEP for every basin pair.
            3. Plot every path on the 2D surface and each 1D arc-length profile.
            4. Assemble the full CTMC rate matrix and summarize pairwise rates.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            from itertools import combinations

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import matplotlib.patheffects as pe
            import stochkin as sk

            from stochkin.plotting import _apply_pub_axes, _apply_pub_cbar
            from stochkin.style import publication_style
            """
        ),
        code(
            """
            _PAIR_COLOURS = [
                "#e6194b",
                "#3cb44b",
                "#4363d8",
                "#f58231",
                "#911eb4",
                "#42d4f4",
                "#f032e6",
                "#bfef45",
                "#fabebe",
                "#469990",
            ]


            def colour_for(index):
                return _PAIR_COLOURS[index % len(_PAIR_COLOURS)]
            """
        ),
        code(
            """
            fes2d_path = DATA_DIR / "synthetic_2d_fes.dat"
            D_s = 0.04
            T = 300.0
            neb_images = 120
            neb_steps = 8000
            max_basins = None
            core_fraction = 0.05

            result = sk.run_multi_mfep_ctmc(
                fes2d_path=fes2d_path,
                D_s=D_s,
                T=T,
                neb_images=neb_images,
                neb_steps=neb_steps,
                max_basins=max_basins,
                core_fraction=core_fraction,
            )

            basin_net = result["basin_network"]
            basins = basin_net.basins
            basin_labels = [f"B{bid}" for bid in result["basin_ids"]]
            pair_keys = sorted(result["legs"].keys())
            """
        ),
        code(
            """
            K_ns = pd.DataFrame(
                result["K_ps"] * 1000.0,
                index=basin_labels,
                columns=basin_labels,
            )
            exit_times_ns = pd.Series(
                result["exit_ps"] / 1000.0,
                index=basin_labels,
                name="exit_time_ns",
            )

            print(f"Detected {len(basins)} basins")
            for basin in basins:
                print(
                    f"Basin {basin.id}: minimum ({basin.minimum[0]:.2f}, {basin.minimum[1]:.2f}), "
                    f"F_min = {basin.f_min:.2f} kJ/mol"
                )
            """
        ),
        code(
            """
            K_ns
            """
        ),
        code(
            """
            rows = []
            for i, j in pair_keys:
                leg = result["legs"][(i, j)]
                path = leg["mfep_path"]
                barrier = float(np.nanmax(path.F - np.nanmin(path.F)))
                K_leg = leg["K"]
                if K_leg.shape[0] >= 2:
                    k_fwd = float(K_leg[0, -1])
                    k_bwd = float(K_leg[-1, 0])
                else:
                    k_fwd = float("nan")
                    k_bwd = float("nan")
                rows.append(
                    {
                        "pair": f"B{basins[i].id}->B{basins[j].id}",
                        "k_fwd": k_fwd,
                        "k_bwd": k_bwd,
                        "barrier_kj_mol": barrier,
                    }
                )

            summary_df = pd.DataFrame(rows)
            summary_df
            """
        ),
        code(
            """
            x_grid, y_grid, fes_grid = sk.load_plumed_fes_2d(fes2d_path, verbose=False)
            kT = result["kT"]

            out_paths = OUT_DIR / "05_all_paths_on_fes.png"
            out_profiles = OUT_DIR / "05_pairwise_profiles.png"
            out_rates = OUT_DIR / "05_rate_matrix.png"
            out_exit = OUT_DIR / "05_exit_times.png"

            with publication_style():
                fig, ax = plt.subplots(figsize=(4.5, 3.6))
                cs = ax.contourf(
                    x_grid,
                    y_grid,
                    (fes_grid / kT).T,
                    levels=np.linspace(0, 15, 30),
                    cmap="rainbow_r",
                )
                cbar = fig.colorbar(cs, ax=ax)
                _apply_pub_cbar(cbar, label=r"$F\\,/\\,k_\\mathrm{B}T$")

                for basin in basins:
                    ax.plot(
                        *basin.minimum,
                        "o",
                        color="white",
                        ms=10,
                        zorder=5,
                        markeredgecolor="black",
                        markeredgewidth=0.6,
                    )
                    ax.text(
                        basin.minimum[0] + 0.15,
                        basin.minimum[1] + 0.15,
                        f"B{basin.id}",
                        color="white",
                        fontsize=10,
                        fontweight="bold",
                        zorder=6,
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                    )

                for index, (i, j) in enumerate(pair_keys):
                    path = result["legs"][(i, j)]["mfep_path"]
                    colour = colour_for(index)
                    ax.plot(
                        path.x,
                        path.y,
                        "-",
                        color=colour,
                        lw=2.0,
                        alpha=0.9,
                        label=f"B{basins[i].id}->B{basins[j].id}",
                        zorder=4,
                    )

                ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
                _apply_pub_axes(ax, r"CV$_1$", r"CV$_2$", "Pairwise MFEPs on 2D FES")
                fig.tight_layout()
                fig.savefig(out_paths, dpi=300)

            n_pairs = len(pair_keys)
            n_cols = min(n_pairs, 3)
            n_rows = int(np.ceil(n_pairs / n_cols))

            with publication_style():
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(3.3 * n_cols, 2.6 * n_rows),
                    squeeze=False,
                )

                for index, (i, j) in enumerate(pair_keys):
                    row, col = divmod(index, n_cols)
                    ax = axes[row, col]
                    leg = result["legs"][(i, j)]
                    path = leg["mfep_path"]

                    s = path.s
                    F = path.F - np.nanmin(path.F)
                    colour = colour_for(index)

                    ax.fill_between(s, F / kT, alpha=0.20, color=colour)
                    ax.plot(s, F / kT, "-", color=colour, lw=1.5)

                    basin_network_1d = leg.get("basin_network")
                    if basin_network_1d is not None:
                        for basin_1d in basin_network_1d.basins:
                            idx_min = np.argmin(np.abs(s - basin_1d.minimum))
                            ax.axvline(s[idx_min], ls=":", color="grey", lw=0.7, alpha=0.6)

                    K_leg = leg["K"]
                    if K_leg.shape[0] >= 2:
                        k_fwd = K_leg[0, -1]
                        k_bwd = K_leg[-1, 0]
                        ax.text(
                            0.97,
                            0.95,
                            f"$k_{{->}}={k_fwd:.2e}$\\n$k_{{<-}}={k_bwd:.2e}$",
                            transform=ax.transAxes,
                            fontsize=6,
                            va="top",
                            ha="right",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        )

                    _apply_pub_axes(
                        ax,
                        xlabel="Arc-length  s",
                        ylabel=r"$F(s) / k_\\mathrm{B}T$",
                        title=f"B{basins[i].id} -> B{basins[j].id}",
                    )

                for index in range(n_pairs, n_rows * n_cols):
                    row, col = divmod(index, n_cols)
                    axes[row, col].set_visible(False)

                fig.tight_layout()
                fig.savefig(out_profiles, dpi=300)

            with publication_style():
                K_abs = np.abs(K_ns.to_numpy())
                with np.errstate(divide="ignore", invalid="ignore"):
                    Klog = np.where(K_abs > 0, np.log10(K_abs), np.nan)

                fig, ax = plt.subplots(figsize=(3.3, 2.8))
                im = ax.imshow(Klog, cmap="magma_r", aspect="auto")
                ax.set_xticks(range(len(basin_labels)))
                ax.set_xticklabels(basin_labels)
                ax.set_yticks(range(len(basin_labels)))
                ax.set_yticklabels(basin_labels)

                finite_klog = Klog[np.isfinite(Klog)]
                threshold = np.nanmedian(finite_klog) if finite_klog.size else 0.0
                for ii in range(len(basin_labels)):
                    for jj in range(len(basin_labels)):
                        value = K_ns.to_numpy()[ii, jj]
                        if ii != jj and value > 0:
                            ax.text(
                                jj,
                                ii,
                                f"{value:.1e}",
                                ha="center",
                                va="center",
                                fontsize=5,
                                color="white" if Klog[ii, jj] < threshold else "black",
                            )

                cbar = fig.colorbar(im, ax=ax)
                _apply_pub_cbar(cbar, label=r"$\\log_{10}|K_{ij}|$  [ns$^{-1}$]")
                _apply_pub_axes(ax, title=f"Rate matrix ({len(basin_labels)} basins)")
                fig.tight_layout()
                fig.savefig(out_rates, dpi=300)

            with publication_style():
                fig, ax = plt.subplots(figsize=(3.3, 2.8))
                bars = ax.bar(
                    range(len(basin_labels)),
                    exit_times_ns.to_numpy(),
                    color=[colour_for(index) for index in range(len(basin_labels))],
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_yscale("log")
                ax.set_xticks(range(len(basin_labels)))
                ax.set_xticklabels(basin_labels)
                for bar, value in zip(bars, exit_times_ns.to_numpy()):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                _apply_pub_axes(ax, xlabel="Basin", ylabel="Exit time  [ns]", title="Mean exit times")
                fig.tight_layout()
                fig.savefig(out_exit, dpi=300)

            print(f"Saved {out_paths.relative_to(ROOT)}")
            print(f"Saved {out_profiles.relative_to(ROOT)}")
            print(f"Saved {out_rates.relative_to(ROOT)}")
            print(f"Saved {out_exit.relative_to(ROOT)}")
            plt.show()
            """
        ),
        code(
            """
            exit_times_ns
            """
        ),
    ],
    "06_uncertainty.ipynb": [
        md(
            """
            # Example 06 - Uncertainty Propagation on a 1D FES

            This notebook mirrors `examples/06_uncertainty.py`. It builds a
            synthetic 1D double-well free-energy surface, perturbs it with
            uncertainty models, and estimates confidence intervals on rates and
            exit times via bootstrap resampling.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            import numpy as np
            import matplotlib.pyplot as plt
            import stochkin as sk

            from stochkin.plotting import _apply_pub_axes
            from stochkin.style import publication_style
            """
        ),
        code(
            """
            def make_synthetic_fes(n=300, barrier=8.0):
                s = np.linspace(0.0, 1.0, n)
                x = 2.0 * s - 1.0
                F = barrier * (1.0 - x ** 2) ** 2
                F -= F.min()
                return s, F


            def make_synthetic_D(s, D0=0.02, amp=0.005):
                return D0 - amp * np.exp(-((s - 0.5) ** 2) / (2 * 0.05 ** 2))
            """
        ),
        code(
            """
            n_bootstrap = 200
            seed = 42

            s, F = make_synthetic_fes()
            D = make_synthetic_D(s)

            res1 = sk.bootstrap_ctmc_1d(
                s,
                F,
                D,
                F_err=0.5,
                D_rel_err=0.3,
                n_bootstrap=n_bootstrap,
                seed=seed,
                T=300.0,
                time_unit="ps",
                verbose=True,
            )

            F_sigma = 0.3 + 0.8 * np.exp(-((s - 0.5) ** 2) / (2 * 0.08 ** 2))
            D_lo = D * np.exp(-0.5)
            D_hi = D * np.exp(+0.5)

            res2 = sk.bootstrap_ctmc_1d(
                s,
                F,
                D,
                F_err=F_sigma,
                D_lo=D_lo,
                D_hi=D_hi,
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
                T=300.0,
                time_unit="ps",
                verbose=True,
            )
            """
        ),
        code(
            """
            print("Scenario 1")
            print(res1.summary("ps"))
            print()
            print("Scenario 2")
            print(res2.summary("ps"))
            """
        ),
        code(
            """
            output_path = OUT_DIR / "06_uncertainty.png"

            with publication_style():
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(9, 6.5),
                    gridspec_kw={"hspace": 0.45, "wspace": 0.38},
                )

                ax = axes[0, 0]
                ax.fill_between(s, F - F_sigma, F + F_sigma, alpha=0.25, color="C0", label="sigma_F(s)")
                ax.plot(s, F, "C0-", lw=1.5, label="F(s)")
                ax.set_xlabel("s")
                ax.set_ylabel("F [kJ/mol]")
                ax.set_title("(a) FES +/- uncertainty")
                ax.legend(fontsize=8)
                _apply_pub_axes(ax)

                ax = axes[0, 1]
                ax.fill_between(s, D_lo, D_hi, alpha=0.25, color="C1", label="D CI")
                ax.plot(s, D, "C1-", lw=1.5, label="D(s)")
                ax.set_xlabel("s")
                ax.set_ylabel("D [CV^2/ps]")
                ax.set_title("(b) Diffusion +/- CI")
                ax.legend(fontsize=8)
                _apply_pub_axes(ax)

                ax = axes[1, 0]
                k01 = res2.K_samples[:, 0, 1]
                ax.hist(k01, bins=30, density=True, alpha=0.7, color="C2", edgecolor="white", linewidth=0.5)
                ax.axvline(res2.K_mean[0, 1], color="k", ls="--", lw=1.2, label=f"mean = {res2.K_mean[0, 1]:.4g}")
                ax.axvline(res2.K_ci_lo[0, 1], color="grey", ls=":", lw=1)
                ax.axvline(res2.K_ci_hi[0, 1], color="grey", ls=":", lw=1, label=f"{res2.confidence_level:.0%} CI")
                ax.set_xlabel("k_0->1 [ps^-1]")
                ax.set_ylabel("density")
                ax.set_title("(c) Rate 0->1 bootstrap")
                ax.legend(fontsize=8)
                _apply_pub_axes(ax)

                ax = axes[1, 1]
                tau0 = res2.exit_mean_samples[:, 0]
                ax.hist(tau0, bins=30, density=True, alpha=0.7, color="C3", edgecolor="white", linewidth=0.5)
                ax.axvline(
                    res2.exit_mean_mean[0],
                    color="k",
                    ls="--",
                    lw=1.2,
                    label=f"mean = {res2.exit_mean_mean[0]:.4g}",
                )
                ax.axvline(res2.exit_mean_ci_lo[0], color="grey", ls=":", lw=1)
                ax.axvline(
                    res2.exit_mean_ci_hi[0],
                    color="grey",
                    ls=":",
                    lw=1,
                    label=f"{res2.confidence_level:.0%} CI",
                )
                ax.set_xlabel("<tau_exit> [ps]")
                ax.set_ylabel("density")
                ax.set_title("(d) Exit time basin 0 bootstrap")
                ax.legend(fontsize=8)
                _apply_pub_axes(ax)

                fig.savefig(output_path, dpi=300, bbox_inches="tight")

            print(f"Saved {output_path.relative_to(ROOT)}")
            plt.show()
            """
        ),
    ],
}


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for name, cells in NOTEBOOK_SPECS.items():
        path = NOTEBOOKS_DIR / name
        path.write_text(json.dumps(notebook(cells), indent=2) + "\n", encoding="utf-8")
        written.append(path.relative_to(ROOT).as_posix())

    print("Wrote notebooks:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
