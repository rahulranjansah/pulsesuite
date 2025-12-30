
import os
import sys
import sphinx_rtd_theme
from datetime import datetime

# -- Project information -----------------------------------------------------

project = 'PulseSuite'
copyright = f'{datetime.now().year}, PulseSuite Developers'
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
    "sphinx_copybutton",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- AutoAPI configuration ---------------------------------------------------
autoapi_type = 'python'
autoapi_dirs = ['../../src']
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_add_toctree_entry = True

# -- myst configuration ---------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "substitution",
    "dollarmath",
    "amsmath",
]

# Configure source file types
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.myst.md': 'markdown',  # Treat as markdown, not jupyter notebook
}
