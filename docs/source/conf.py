# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Imports -----------------------------------------------------------------
from datetime import date
from numpydoc import xref

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyXC"
copyright = f"2022 - {date.today().year}, Sang-Hyeok Lee"
author = "Sang-Hyeok Lee"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "nbsphinx",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

autoapi_dirs = ["../../pyxc"]
autoapi_type = "python"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autodoc_mock_imports = ["pyxc.layer.Layer"]

html_css_files = [
    "lb_signatures.css",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
default_role = "autolink"


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# numpydoc settings
numpydoc_xref_ignore = {"optional", "type_without_description", "BadException"}
# numpydoc_validation_checks = {"all"}
numpydoc_xref_param_type = True
numpydoc_xref_aliases = xref.DEFAULT_LINKS

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_copy_source = False
html_domain_indices = True

# -- For long long types -----------------------------------------------------
from numpy.typing import ArrayLike

autodoc_type_aliases = {
    "ArrayLike": "array_like",
}

# -- Intersphinx -------------------------------------------------------------
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/devdocs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- NBSphinx -----------------------------------------------------------------
nbsphinx_allow_errors = True
