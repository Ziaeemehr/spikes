import sys
import os
from unittest.mock import MagicMock as Mock
from setuptools_scm import get_version

# sys.path.insert(0,os.path.abspath("../examples"))
sys.path.insert(0,os.path.abspath("../spikes"))
sys.path.insert(0, os.path.abspath("../examples"))
sys.path.insert(0, os.path.abspath(".."))

needs_sphinx = '1.3'

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx.ext.graphviz',
	'sphinx.ext.viewcode'
]

# Exclude notebook output cells
nbsphinx_execute = 'never' # to avoid running notebook for document
# Set timeout for cell execution (in seconds)
nbsphinx_timeout = 300

source_suffix = '.rst'
master_doc = 'index'
project = u'spikes'
copyright = u'2024, Abolfazl Ziaeemehr'

release = version = get_version(root='..', relative_to=__file__)

default_role = "any"
add_function_parentheses = True
add_module_names = False
html_theme = 'nature'
pygments_style = 'colorful'
# htmlhelp_basename = 'JiTCODEdoc'
exclude_patterns = ['_build', '**.ipynb_checkpoints']

numpydoc_show_class_members = False
autodoc_member_order = 'bysource'
graphviz_output_format = "svg"
toc_object_entries_show_parents = 'hide'

def on_missing_reference(app, env, node, contnode):
	if node['reftype'] == 'any':
		return contnode
	else:
		return None

def setup(app):
	app.connect('missing-reference', on_missing_reference)
 
# nbsphinx_prolog = """
# {% set docname = env.doc2path(env.docname, base=None) %}
# .. note::
#    This page was generated from `{{ docname }}`__.

#    __ https://github.com/Ziaeemehr/spikes/blob/main/{{ docname }}
# """