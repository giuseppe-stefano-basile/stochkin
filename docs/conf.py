# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root so autodoc can find the package.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "stochkin"
author = "Giuseppe Stefano Basile"
copyright = "2024–2026, Giuseppe Stefano Basile"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",          # pull docstrings from code
    "sphinx.ext.napoleon",         # understand NumPy/Google-style docstrings
    "sphinx.ext.mathjax",          # render LaTeX math
    "sphinx.ext.viewcode",         # add [source] links to highlighted code
    "sphinx.ext.intersphinx",      # cross-reference NumPy/SciPy docs
    "sphinx.ext.autosummary",      # generate summary tables
]

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Autosummary — we use explicit module pages under api/ so disable
# automatic stub generation to avoid duplicate-object warnings from
# dataclass fields.
autosummary_generate = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Suppress warnings for optional imports that may not be installed
# when building the docs (fipy, tqdm, pandas, …).
autodoc_mock_imports = ["fipy", "tqdm", "pandas"]

# Suppress harmless warnings:
# - Cross-reference ambiguities for common names like n_basins.
suppress_warnings = ["ref.python"]

# Use only the class docstring (not __init__) for class documentation.
autodoc_class_content = "class"


# ---- Filter duplicate-object warnings from dataclass fields ----------------
# Napoleon's Attributes section and autodoc's member introspection both
# generate ``.. attribute::`` directives for @dataclass fields, which
# triggers a "duplicate object description" warning from the Python domain.
# These are harmless — we install a logging filter to silence them.
import logging as _logging


class _DuplicateObjectFilter(_logging.Filter):
    """Suppress 'duplicate object description' warnings."""

    def filter(self, record: _logging.LogRecord) -> bool:
        return "duplicate object description" not in record.getMessage()


# Apply the filter as early as possible (module load time).
for _handler in _logging.getLogger("sphinx").handlers:
    _handler.addFilter(_DuplicateObjectFilter())
# Also add directly to the domain logger and the root sphinx logger.
_logging.getLogger("sphinx").addFilter(_DuplicateObjectFilter())
_logging.getLogger("sphinx.domains.python").addFilter(_DuplicateObjectFilter())


def setup(app):
    # Re-apply once more after Sphinx has set up its own logging.
    for handler in _logging.getLogger("sphinx").handlers:
        handler.addFilter(_DuplicateObjectFilter())
    _logging.getLogger("sphinx").addFilter(_DuplicateObjectFilter())
    _logging.getLogger("sphinx.domains.python").addFilter(_DuplicateObjectFilter())

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"  # clean, modern theme (pip install furo)
html_title = "stochkin"
html_static_path = ["_static"]

# Furo-specific options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "preamble": r"\usepackage{amsmath,amssymb}",
}

# -- Exclude patterns --------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Source suffix ------------------------------------------------------------
source_suffix = ".rst"
master_doc = "index"
