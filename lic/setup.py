from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    name = "lic_internal",
    version = "1.0",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("lic_internal", ["lic_internal.pyx"],
                             include_dirs = [np.get_include()])]
)
