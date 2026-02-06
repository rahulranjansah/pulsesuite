#!/usr/bin/env python3
"""
PulseSuite documentation build configuration file.
"""

from importlib.metadata import version as pkg_version
import os
import sys
import sphinx_rtd_theme
from datetime import datetime

# -- Project information -----------------------------------------------------

project = "pulsesuite"
copyright = f"{datetime.now().year}, Rahul R. Sah and PulseSuite Developers"
author = "PulseSuite Developers"

# Make pulsesuite importable for executed notebooks and autodoc
# docs/source/conf.py -> ../../src is correct for your repo layout
sys.path.insert(0, os.path.abspath("../../src"))

# Get version from package metadata
project_ver = pkg_version(project)
version = ".".join(project_ver.split(".")[:2])
release = project_ver

# -- General configuration ---------------------------------------------------

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.load_style",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.mathjax",  # Maths visualization
    "sphinx.ext.graphviz",  # Dependency diagrams
    "sphinx_copybutton",
    "notfound.extension",
    "hoverxref.extension",
    "sphinx_github_role",
    "myst_nb",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc configuration
autodoc_member_order = "bysource"

# MathJax configuration
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
mathjax2_config = {
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "processEscapes": True,
        "ignoreClass": "document",
        "processClass": "math|output_area",
    }
}
myst_update_mathjax = False

# Myst configuration
myst_enable_extensions = [
    "colon_fence",
    "substitution",
    "dollarmath",
    "amsmath",
]

myst_substitutions = {
    # "SBEs": "{py:class}`~pulsesuite.PSTD3D.SBEs`",

}

# Hoverxref Extension
hoverxref_auto_ref = True
hoverxref_mathjax = True
hoverxref_intersphinx = [
    "numpy",
    "scipy",
    "matplotlib",
    "numba",
    "pyfftw",
]
hoverxref_domains = ["py"]
hoverxref_role_types = {
    "hoverxref": "modal",
    "ref": "modal",
    "confval": "tooltip",
    "mod": "tooltip",
    "class": "tooltip",
    "meth": "tooltip",
    "obj": "tooltip",
}

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Source file types
# IMPORTANT: Use "myst" as the parser; myst_nb extension handles execution
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".myst.md": "myst-nb",
}

# Warning suppression
suppress_warnings = [
    "image.nonlocal_uri",
    "autoapi.python_import_resolution",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]

# -- MyST-NB execution configuration ----------------------------------------

# Execute notebooks (i.e., {code-cell} blocks)
# "force" means always execute, "cache" means use cached outputs if available
nb_execution_mode = "force"
nb_execution_timeout = 600  # Increased to 10 minutes for heavy computations like InitializeSBE

# Allow errors to be displayed (set False to fail build on errors)
nb_execution_allow_errors = True

# Show traceback on errors
nb_execution_raise_on_error = False

# Show stderr output
nb_execution_show_tb = True

# Skip execution of specific patterns if they cause kernel crashes
# Add files that consistently cause DeadKernelError
# Note: If a file causes kernel crashes even with try-except blocks,
# it likely means the crash happens at a lower level (C extensions, memory issues)
# and the file should be excluded from execution
nb_execution_excludepatterns = [
    # "examples/sbes_example.myst.md",
    # "examples/coulomb_example.myst.md" # Excluded due to DeadKernelError during heavy computations
]

# Your .myst.md files declare format_name: myst in the YAML header,
# so if you keep jupytext, the fmt must be "myst" (not "mystnb")
nb_custom_formats = {
    ".myst.md": ("jupytext.reads", {"fmt": "myst"}),
}

# -- AutoAPI configuration ---------------------------------------------------

autoapi_type = "python"
autoapi_dirs = ["../../src"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_add_toctree_entry = True
autoapi_ignore = [
    "*_compat*",
]

exclude_patterns.extend(["autoapi/index.rst", "autoapi/pulsesuite/index.rst"])

# Pygments style
pygments_style = "sphinx"

# Ignore sphinx-autoapi warnings on reimported objects
suppress_warnings.append("autoapi.python_import_resolution")

latex_engine = "xelatex"

latex_elements = {
    "fontpkg": r"""
\setmainfont{FreeSerif}[
  UprightFont    = *,
  ItalicFont     = *Italic,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldItalic
]
\setsansfont{FreeSans}[
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]
\setmonofont{FreeMono}[
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]
""",
}

bibtex_bibfiles = ["pulsesuite.bib"]
