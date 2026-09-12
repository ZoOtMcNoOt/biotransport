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
napoleon_preprocess_types = True
napoleon_type_aliases = {"callable": "collections.abc.Callable"}
# Render an "Attributes:" section as :ivar: fields rather than separate object
# descriptions. Without this, a class that both documents its attributes in the
# docstring and annotates them for type checking gets each one described twice.
napoleon_use_ivar = True

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
# Only mock the compiled extension when it genuinely is not importable.
#
# Mocking it unconditionally is worse than it looks: every pybind11 class becomes
# an empty stand-in, so the whole native API renders with a heading and no
# members, and the build still reports success. When the package is installed --
# which it is in CI and in any editable dev checkout -- autodoc should import the
# real module and document the real signatures.
try:  # pragma: no cover - documentation build only
    import biotransport._core._core  # noqa: F401

    autodoc_mock_imports = []
    _native_available = True
except ImportError:  # pragma: no cover - documentation build only
    autodoc_mock_imports = ["biotransport._core._core"]
    _native_available = False
    suppress_warnings = ["autodoc.mocked_object"]
    print(
        "conf.py: the compiled extension is not importable, so the native API "
        "reference will be mocked and incomplete. Build it with "
        "`pip install -e .` for complete docs."
    )

def _escape_rst_substitutions(app, what, name, obj, options, lines):
    """Stop maths notation in docstrings from being parsed as RST markup.

    Docstrings across the project write magnitudes the way the textbooks do --
    ``|dR/dc|``, ``|G*|`` -- and reStructuredText reads anything between vertical
    bars as a substitution reference, then errors because no such substitution
    exists. Several of these live in the compiled pybind11 docstrings, which
    cannot be edited from Python, so the rewrite happens here at render time.
    """

    import re as _re

    # Escape bare delimiters instead of inserting inline literals. A magnitude
    # can touch the rest of a formula ("|z|FD"), where closing literal markup is
    # invalid. Preserve bars already escaped by the author and code literals.
    pattern = _re.compile(r"(?<![\\`])\|([^|\s`][^|`]*?)(?<!\\)\|(?!`)")
    for index, line in enumerate(lines):
        if "|" in line:
            lines[index] = pattern.sub(r"\\|\1\\|", line)


def _public_native_signature(app, what, name, obj, options, signature, return_annotation):
    """Render native list returns without synthetic pybind metadata.

    ``FixedSize`` in pybind's ``Annotated`` signature is not an importable
    Python type. These list lengths and orderings are documented in api/core.rst.
    Keep the actual Python return type instead of linking that metadata as a
    class or hiding other unresolved native annotations.
    """

    if name.endswith(
        (".TransportProblem.boundaries", ".StructuredMesh3D.ijk")
    ) and return_annotation:
        import re as _re

        normalized = _re.sub(
            r"^Annotated\[(list\[.+\]), FixedSize\(\d+\)\]$",
            r"\1",
            return_annotation,
        )
        if normalized != return_annotation:
            return signature, normalized
    return None


def setup(app):  # pragma: no cover - documentation build only
    app.connect("autodoc-process-docstring", _escape_rst_substitutions)
    app.connect("autodoc-process-signature", _public_native_signature)
    return {"parallel_read_safe": True}


# Surface broken cross-references instead of letting them pass silently. The
# ignore list below covers references that genuinely cannot resolve -- pybind11
# base classes, third-party types without inventories, and private helpers -- so
# that whatever is left is a real broken link worth fixing.
nitpicky = True
nitpick_ignore_regex = [
    # pybind11 synthesises this base class for every bound type.
    (r"py:class", r"pybind11_builtins\..*"),
    # Compiled types that autodoc reaches through annotations but does not render.
    (r"py:class", r"biotransport\._core\..*"),
    # Private helpers and type variables, deliberately undocumented.
    (r"py:class", r".*\._[A-Z_].*"),
    (r"py:class", r"JSONValue"),
    # Third parties without a resolvable inventory entry for these names.
    (r"py:class", r"numpy\..*"),
    (r"py:class", r"np\..*"),
    (r"py:class", r"matplotlib\..*"),
    (r"py:class", r"mpl_toolkits\..*"),
    (r"py:class", r"ArrayLike"),
    (r"py:class", r"Axes.*"),
    (r"py:class", r"Figure"),
    (r"py:class", r"SubFigure"),
    (r"py:class", r"FuncAnimation"),
    (r"py:class", r"optional"),
]
