# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import sys
import os
sys.path.insert(0, os.path.abspath('../../src/python/double_pendulum/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Double Pendulum'
copyright = '2022, Underactuated Lab'
author = 'Underactuated Lab'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'numpydoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              "sphinx.ext.autosectionlabel",
              ]
numpydoc_show_class_members = False


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options={'titles_only' : True,
                    'collapse_navigation' : False,
                    'navigation_depth' : 5,
                    }
html_static_path = ['_static']

#-----------------------------------------------------------------------------
#html_logo = f'../figures/logo.jpg'
html_logo = f'../figures/chaotic_freefall_long_exposure_shot.jpg'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
}
