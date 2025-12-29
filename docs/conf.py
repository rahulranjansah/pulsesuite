
import os
import sys
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'PulseSuite'
copyright = '2025, PulseSuite Developers'
author = 'PulseSuite Developers'
release = '0.1.0'
version = '0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- AutoAPI configuration ---------------------------------------------------
autoapi_type = 'python'
autoapi_dirs = ['../src', '../PSTD3D']
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_add_toctree_entry = True
