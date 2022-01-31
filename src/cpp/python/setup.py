from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ilqr_extension = Extension(
    name="cppilqr",
    sources=["cppilqr.pyx"],
    libraries=["ilqr", "yaml-cpp"],
    library_dirs=["../controllers/ilqr/obj"],
    include_dirs=["../controllers/ilqr/obj"],
    language="c++"
)
setup(
    name="cppilqr",
    ext_modules=cythonize([ilqr_extension]),
)
