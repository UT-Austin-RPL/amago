# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AMAGO'
#copyright = '2025, UT Austin Robot Perception and Learning Lab'
author = 'UT Austin Robot Perception and Learning Lab'
html_title = "AMAGO"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    "sphinx.ext.viewcode",
    'sphinx_autodoc_typehints',
]

master_doc = "index"
napoleon_google_docstring = True
myst_enable_extensions = ["colon_fence"]
napoleon_numpy_docstring = False
napoleon_custom_sections = [
    ("Environment", "params"),
    ("Logging", "params"),
    ("Replay", "params"),
    ("Learning Schedule", "params"),
    ("Optimization", "params"),
]

autosummary_generate = True
add_module_names = False
modindex_common_prefix = ["amago."]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "private-members": False,
    "special-members": "__init__",
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = [os.path.abspath(os.path.join('..', 'examples')), "media"]

html_logo = "media/amago_logo_3.png"

html_css_files = [
    'custom.css',
]

import sphinx_book_theme
html_theme = "sphinx_book_theme"

html_theme_options = {
    "default_mode": "light",
}

html_context = {
    "default_mode": "light",
}
