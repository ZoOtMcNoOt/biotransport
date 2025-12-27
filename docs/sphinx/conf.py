# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add Python package to path
sys.path.insert(0, os.path.abspath("../../python"))

# -- Project information -----------------------------------------------------
project = "BioTransport"
copyright = "2025, BioTransport Authors"
author = "BioTransport Authors"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autosummary settings
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# MyST parser for Markdown support
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "BioTransport"
html_static_path = ["_static"]
html_css_files = []

# Theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2980b9",
        "color-brand-content": "#2980b9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#56b4e9",
        "color-brand-content": "#56b4e9",
    },
}

# -- Options for autodoc -----------------------------------------------------
# Mock imports for C++ extension module
autodoc_mock_imports = ["biotransport._core"]

# Suppress warnings for mocked objects (C++ extension not available during doc build)
suppress_warnings = [
    "autodoc.mocked_object",
    "autodoc",  # Suppress all autodoc warnings for mocked C++ extension
]
