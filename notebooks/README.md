# Example notebooks

This directory contains Jupyter notebook versions of the repository examples.
They are intended to mirror the workflows in `examples/` while being easier to
run interactively from Jupyter.

## Files

- `00_generate_synthetic_data.ipynb`
- `01_analytic_doublewell.ipynb`
- `02_1d_plumed_fes_ctmc.ipynb`
- `03_1d_hummer_D_ctmc.ipynb`
- `04_mfep_ctmc.ipynb`
- `05_pairwise_mfep_paths.ipynb`
- `06_uncertainty.ipynb`

## Regeneration

The notebooks are generated from templates in
`tools/build_example_notebooks.py`.

To rebuild them:

```bash
python tools/build_example_notebooks.py
```

Notebook-generated figures are written to `notebooks/output/`, which is
gitignored.
