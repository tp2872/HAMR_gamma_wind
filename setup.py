#!/usr/bin/python
#usage:
#python setup.py build_ext --inplace
import os
import numpy as np
import sys
from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#do not do any parallel processing if it is Mac OS X
if sys.platform == "darwin":
    setup(
        cmdclass={'build_ext': build_ext},
        ext_modules=[Extension("pp_c", sources=["pp_c.pyx", "functions.c"], include_dirs=[np.get_include()], extra_link_args=["-O2"])]
        )
else:
    setup(
        cmdclass={'build_ext': build_ext},
        ext_modules=[Extension("pp_c", sources=["pp_c.pyx", "functions.c"], include_dirs=[np.get_include()], extra_compile_args=["-fopenmp"], extra_link_args=["-O2 -fopenmp"])]
        )
