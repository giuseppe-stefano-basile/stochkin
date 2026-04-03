# Changelog

All notable changes to **stochkin** will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] – 2026-03-29

### Added
- `stochkin.uncertainty` module with:
  - `bootstrap_ctmc_1d()` — Monte Carlo bootstrap propagation of F(s)
    and D(s) credible intervals through the full CTMC pipeline.
  - `bootstrap_ctmc_with_hummer_D()` — convenience wrapper that reads
    Hummer-pipeline CSV files directly.
  - `UncertaintyResult` dataclass with `*_mean`, `*_std`, `*_ci_lo`,
    `*_ci_hi` for every CTMC output (K, exit times, rates, branching).
- Example `06_uncertainty.py` — two-scenario demonstration with
  publication-quality figure output.
- Sphinx API page `docs/api/uncertainty.rst`.

### Fixed
- `_build_core_labels` now uses the basin object's detected minimum
  position (`basin.minimum`) instead of `np.argmin(F)` within the
  basin.  Previously, for shallow basins on a barrier flank, the core
  could be placed at the basin boundary (lowest absolute F) rather
  than at the local dip, causing exit-time estimates to be off by
  orders of magnitude.

---

## [0.1.0] – 2026-03-27

### Added
- `stochkin.workflows` module with four high-level one-call wrappers:
  - `run_1d_ctmc` – compute CTMC kinetics from arrays `s`, `F`, `D`
  - `run_1d_ctmc_from_plumed` – load a 1D PLUMED FES + constant D → CTMC
  - `run_1d_ctmc_with_hummer_D` – load FES + Hummer Bayesian D(s) → CTMC
  - `run_mfep_ctmc` – 2D FES → MFEP → 1D arc-length → CTMC
- Helper utilities `interface_to_centers` and `interpolate_D_to_grid`
  now public in `stochkin.workflows` (previously only in the CLI script).
- `__version__ = "0.1.0"` exposed at package level.
- `pyproject.toml` for `pip install` support.
- `examples/` directory with four standalone scripts.
- `tests/test_smoke.py` basic smoke tests.

### Changed
- Package renamed from `Stochastic_Estimation` to **`stochkin`**
  (PEP 8 compliant, shorter, more descriptive).
- Module-level docstring updated to reflect the new name and the added
  `workflows` submodule.

---

<!-- [unreleased]: https://github.com/giuseppe-stefano-basile/stochkin/compare/v0.1.0...HEAD -->
<!-- [0.1.0]: https://github.com/giuseppe-stefano-basile/stochkin/releases/tag/v0.1.0 -->
