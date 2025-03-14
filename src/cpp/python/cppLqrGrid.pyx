# distutils: language = c++
# cython: language_level=3

import numpy as np
#from eigency.core cimport *
from cpython cimport PyObject, Py_INCREF#, PyArray_SetBaseObject
from libcpp.vector cimport vector
from libc.stdlib cimport free

# from cppilqr cimport ilqr, ilqr_mpc
from cppLqrGrid cimport Grid4D2d

cimport numpy as np
np.import_array()


cdef class cppGrid4D2d:
    cdef Grid4D2d *grid4D2d

    def __cinit__(self, filename):
        self.grid4D2d = new Grid4D2d(filename)
    def index(self, double x0, double x1, double x2, double x3):
        return self.grid4D2d.index(x0, x1, x2, x3)
        # bint in_bounds(double x0, double x1, double x2, double x3)
