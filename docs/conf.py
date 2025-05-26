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
import inspect

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import inspect
import gin


class _GIN_REQUIRED:
    """
    gin.REQUIRED is just a unique object() put in global. It renders as <object object> in the docstring.
    """
    def __repr__(self):
        return "gin.REQUIRED"

    def __str__(self):
        return "gin.REQUIRED"

# override gin.REQUIRED to render as gin.REQUIRED in the docstring
gin.REQUIRED = _GIN_REQUIRED()

# override the env vars to render in the docstring
os.environ["AMAGO_WANDB_PROJECT"] = "os.environ['AMAGO_WANDB_PROJECT']"
os.environ["AMAGO_WANDB_ENTITY"] = "os.environ['AMAGO_WANDB_ENTITY']"


def mark_gin_configurable(app, what, name, obj, options, lines):
    """
    If this is a class and is gin.configurable, prepend a note to its docstring.

    Also hacky patch for how gin.REQUIRED seems to be rendered as <object object> in the docstring.
    """

    for configurable in gin.config._REGISTRY._selector_map.values():
        if configurable.wrapped == obj:
            break
    else:
        return

    gin_warning = [
        ".. tip::",
        "",
        f"  This {what} is ``@gin.configurable``. Default values of kwargs can be overridden using `gin <https://github.com/google/gin-config>`_.",
        "",
    ]
    for gin_line in reversed(gin_warning):
        lines.insert(2, gin_line)


def setup(app):
    # register our hook on every autodoc docstring
    app.connect("autodoc-process-docstring", mark_gin_configurable)
    # you can also add a little CSS to make `.admonition.gin-configurable` purple
    app.add_css_file("custom.css")

extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    "sphinx.ext.viewcode",
]

master_doc = "index"
default_role = "py:obj"
napoleon_google_docstring = True
myst_enable_extensions = ["colon_fence"]
napoleon_numpy_docstring = False
always_use_bars_union = True
napoleon_include_admonition = True
napoleon_use_parameter_type = True  # Needed if using autodoc_typehints
napoleon_use_keyword = False
# Tell autosummary to actually generate the stub pages
autosummary_generate = True
# Overwrite old stubs whenever you rebuild
autosummary_generate_overwrite = True
# (Optional) hide function/class signatures in the summary table
autosummary_imported_members = False
autodoc_property_type = True

napoleon_custom_sections = [
    ("Gin Configurable Keyword Args", "Args"),
    ("Keyword Args", "Args"),
]
add_module_names = False
modindex_common_prefix = ["amago."]

autosummary_generate = True
add_module_names = False
modindex_common_prefix = ["amago."]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "private-members": False,
    #"special-members": "__init__",
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = [os.path.abspath(os.path.join('..', 'examples')), "media", "_static"]

html_logo = "media/amago_logo_3.png"

html_css_files = [
    'custom.css',
]

import sphinx_book_theme
html_theme = "sphinx_book_theme"

html_context = {
    "default_mode": "light",
}