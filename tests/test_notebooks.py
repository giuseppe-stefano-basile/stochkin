"""Structural checks for the generated example notebooks."""

from __future__ import annotations

import json
from pathlib import Path


EXPECTED_NOTEBOOKS = [
    "00_generate_synthetic_data.ipynb",
    "01_analytic_doublewell.ipynb",
    "02_1d_plumed_fes_ctmc.ipynb",
    "03_1d_hummer_D_ctmc.ipynb",
    "04_mfep_ctmc.ipynb",
    "05_pairwise_mfep_paths.ipynb",
    "06_uncertainty.ipynb",
]


def test_generated_notebooks_exist_and_have_cells():
    root = Path(__file__).resolve().parents[1]
    notebooks_dir = root / "notebooks"

    for name in EXPECTED_NOTEBOOKS:
        path = notebooks_dir / name
        assert path.exists(), f"Missing notebook: {path}"

        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"], f"Notebook has no cells: {path}"
        assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
        assert any(cell["cell_type"] == "markdown" for cell in notebook["cells"])

        for idx, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{path.name}::cell_{idx}", "exec")
