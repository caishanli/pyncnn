from setuptools import setup, find_packages

import sys
if sys.version_info < (3,0):
  sys.exit('Sorry, Python < 3.0 is not supported')

setup(
  name          = 'pyncnn',
  version       = '${PACKAGE_VERSION}',
  packages      = [ 'pyncnn' ],
  url           = 'https://github.com/caishanli/pyncnn',
  package_dir   = {'': '${CMAKE_CURRENT_BINARY_DIR}'},
  package_data  = {'': ['pyncnn${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}']}
)
