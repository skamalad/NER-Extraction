from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

class BuildExt(build_ext):
    def build_extensions(self):
        # Add compilation flags
        if sys.platform == 'darwin':  # macOS
            for ext in self.extensions:
                ext.extra_compile_args = ['-O3', '-march=native']
        elif sys.platform == 'linux':  # Linux
            for ext in self.extensions:
                ext.extra_compile_args = ['-O3', '-march=native']
        build_ext.build_extensions(self)

setup(
    name='fast_ner',
    version='0.1',
    ext_modules=[
        Extension(
            'libner',
            sources=['src/ner.c'],
            include_dirs=['src'],
        ),
    ],
    cmdclass={'build_ext': BuildExt},
    packages=['fast_ner'],
    package_dir={'fast_ner': 'src'},
    install_requires=[
        'numpy>=1.19.0',
        'torch>=1.7.0',
        'transformers>=4.0.0',
    ],
)