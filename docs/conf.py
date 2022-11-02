#!/usr/bin/env python3

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

import mock

# from unittest import mock
import basedet

sys.path.insert(0, os.path.abspath("../../"))
for m in [
    # "megengine",
    # "megengine.tensor",
    # "megengine.Tensor",
    # "collections",
]:
    sys.modules[m] = mock.Mock(name=m)

# -- Project information -----------------------------------------------------

project = "BaseDet"
copyright = "2021, Megvii"
author = "BaseDet contributor"

# The full version, including alpha/beta/rc tags

# The short X.Y version

version = basedet.__version__
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    # "sphinxcontrib.autoyaml",
]

intersphinx_mapping = {
    # TODO: 443 for internal mge doc, need to cache objects.inv before building doc
    "megengine": ("https://megengine.org.cn/doc/stable/zh", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "python": ("https://docs.python.org/3", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en_US"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitLab",
            "url": "https://github.com/megvii-research/basedet",
            "icon": "fab fa-gitlab",
        }
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

master_doc = "index"
source_suffix = ['.rst', '.md']


def autodoc_skip_member(app, what, name, obj, skip, options):
    # we hide something deliberately
    if getattr(obj, "__HIDE_SPHINX_DOC__", False):
        return True

    # Hide some that are deprecated or not intended to be used
    HIDDEN = {
        "registers",
    }
    try:
        if name in HIDDEN or (
            hasattr(obj, "__doc__") and obj.__doc__.lower().strip().startswith("deprecated")
        ):
            print("Skipping deprecated object: {}".format(name))
            return True
    except Exception:
        pass
    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
