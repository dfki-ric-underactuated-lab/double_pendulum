# distutils: language = c++
# cython: language_level=3

import numpy as np
#from eigency.core cimport *
from cpython cimport PyObject, Py_INCREF#, PyArray_SetBaseObject
from libcpp.vector cimport vector
from libc.stdlib cimport free
from libcpp.string cimport string
# from cppLqrGrid cimport regular_grid_t

cimport numpy as np
np.import_array()

# cdef extern from "<cube_cell_t>" namespace "prx::utilities":
#      cdef cppclass cell4D2d "prx::utilities::cube_cell_t<Eigen::Vector<double,2>, 4>":

cdef extern from "../controllers/lqr_grid/regular_grid.hpp" namespace "prx::utilities":
    cdef cppclass Grid4D2d "prx::utilities::regular_grid_t<prx::utilities::cube_cell_t<Eigen::Vector<double, 2>, 4>, 4>":
        Grid4D2d(string) except +
        int index(double x0, double x1, double x2, double x3)
        bint in_bounds(double x0, double x1, double x2, double x3)
        # CellType::Element& query(Ts... xs)

