# Changelog

All notable changes to **stochkin** will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

<!-- [unreleased]: https://github.com/giuseppe-invernizzi/stochkin/compare/v0.1.0...HEAD -->
<!-- [0.1.0]: https://github.com/giuseppe-invernizzi/stochkin/releases/tag/v0.1.0 -->
