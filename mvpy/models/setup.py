import numpy as np
import glob
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

if os.name=='nt':
    ext = Extension('_psparse',
        ['psparse.pyx', 'cs_gaxpy.c'],
        extra_compile_args=['/openmp', '/O2', '/Ot', '/fp:precise'],
        include_dirs = [np.get_include(), '.'])
else:
    ext = Extension('_psparse',
        ['psparse.pyx', 'cs_gaxpy.c'],
        extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
        include_dirs = [np.get_include(), '.'],
        extra_link_args=['-fopenmp'])

    
setup(
    cmdclass = {'build_ext': build_ext},
    py_modules = ['psparse',],
    ext_modules = cythonize([ext])
)