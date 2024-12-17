# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Add the `docs` directory to sys.path so Sphinx knows where to find the configuration
sys.path.append(os.path.abspath('docs'))

sys.path.append(os.path.abspath('../packages/augi'))



# -- Project information -----------------------------------------------------

project = 'eazyml'
copyright = '2024, eazyml'
author = 'eazyml'

# The full version, including alpha/beta/rc tags
version  = '0.1'
release = '0.1.1'

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
os.environ["READTHEDOCS_LANGUAGE"] = "en"
os.environ["READTHEDOCS_VERSION"] = "latest"
os.environ["READTHEDOCS_VERSION_NAME"] = "latest"

# Mock missing modules or dependencies
autodoc_mock_imports = ['utility', 'eazyml', 'eazyml_augi']

# Optional: Build Documentation Without Importing Modules: 
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'ignore-module-all': True
}

# Ensure autodoc Doesn’t Raise Errors
suppress_warnings = ['autodoc.import']

# html_js_files = [
#      ("readthedocs.js", {"defer": "defer"}),
# ]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # "sphinx.ext.duration",
    # "sphinx.ext.doctest",
    # "sphinx.ext.todo",
    # "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    # "sphinx.ext.intersphinx",
    # "myst_parser"
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Hide view page source link at top
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']

autodoc_mock_imports = []

html_show_copyright = True

html_show_sphinx = True

# -- Build path override for Read the Docs -----------------------------------

# Force Sphinx to look in the `docs` directory for source files
if os.environ.get("READTHEDOCS"):
    # Redirect Sphinx to build from the "docs" directory
    os.environ["SPHINXBUILD"] = "python -m sphinx -T -W --keep-going -b html -d _build/doctrees -D language=en docs $READTHEDOCS_OUTPUT/html"

# Mock imports for external dependencies (if needed)
autodoc_mock_imports = []