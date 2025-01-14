from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os

# check if running inside conda/mamba environment
conda_prefix = os.getenv("CONDA_PREFIX")
if conda_prefix is not None:
    # use conda environment include directories
    include_dirs = [
        f"{conda_prefix}/include/eigen3",
        f"{conda_prefix}/include",
    ]
else:
    # use default include directories
    include_dirs = []

ilqr_extension = Extension(
    name="cppilqr",
    sources=["cppilqr.pyx"],
    libraries=["ilqr", "yaml-cpp"],
    library_dirs=["../controllers/ilqr/obj"],
    include_dirs=[
        "../controllers/ilqr/obj",
        np.get_include(),
    ]
    + include_dirs,
    language="c++",
)
setup(
    name="cppilqr",
    ext_modules=cythonize([ilqr_extension]),
)
