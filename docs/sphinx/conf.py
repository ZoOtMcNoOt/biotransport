# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add Python package to path
sys.path.insert(0, os.path.abspath("../../python"))

# -- Project information -----------------------------------------------------
project = "BioTransport"
copyright = "2026, BioTransport Authors"
author = "BioTransport Authors"
version = "0.2.0"
release = "0.2.0"

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

# Intersphinx mapping. Set BIOTRANSPORT_DOCS_OFFLINE=1 to build without network
# access; unreachable inventories otherwise emit warnings that fail a -W build.
if os.environ.get("BIOTRANSPORT_DOCS_OFFLINE"):
    intersphinx_mapping = {}
else:
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

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "BioTransport"
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
# The compiled extension is imported for real so native docstrings and
# signatures appear in the rendered API; build the package before the docs.
# Set BIOTRANSPORT_DOCS_MOCK_CORE=1 to fall back to mocking when no build is
# available (native members then render without signatures).
if os.environ.get("BIOTRANSPORT_DOCS_MOCK_CORE"):
    autodoc_mock_imports = ["biotransport._core._core"]
    suppress_warnings = ["autodoc.mocked_object"]
