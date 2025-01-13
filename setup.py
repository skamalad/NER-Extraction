from setuptools import setup, Extension
import numpy as np
import sys

# Define the extension module
module = Extension('libner',
                  sources=['src/ner.c'],
                  include_dirs=[np.get_include(), 'src', sys.prefix + '/include'],
                  extra_compile_args=['-O0', '-g', '-Wall', '-fPIC'],  # Debug build
                  extra_link_args=['-fPIC'])

# Run setup
setup(name='libner',
      version='1.0',
      description='Fast NER implementation in C',
      ext_modules=[module])