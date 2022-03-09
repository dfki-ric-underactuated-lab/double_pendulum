from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ilqr_extension = Extension(
    name="cppilqr",
    sources=["cppilqr.pyx"],
    libraries=["ilqr", "yaml-cpp"],
    library_dirs=["../controllers/ilqr/obj"],
    include_dirs=["../controllers/ilqr/obj", np.get_include()],
    language="c++"
)
setup(
    name="cppilqr",
    ext_modules=cythonize([ilqr_extension]),
)
