# distutils: language = c++
# distutils: sources = ../controllers/ilqr/src/ilqr.cpp ../controllers/ilqr/src/ilqr_mpc.cpp
# cython: language_level=3

import numpy as np
#from eigency.core cimport *
from cpython cimport PyObject, Py_INCREF#, PyArray_SetBaseObject
from libcpp.vector cimport vector
from libc.stdlib cimport free

from cppilqr cimport ilqr, ilqr_mpc

cimport numpy as np
np.import_array()

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr):
        """ Set the data of the array
        This cannot be done in the constructor as it must recieve C-level
        arguments.
        Parameters:
        -----------
        size: int
            Length of the array.
        data_ptr: void*
            Pointer to the data            
        """
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        #cdef np.npy_intp shape[1]
        #shape[0] = <np.npy_intp> self.size
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_DOUBLE, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        #free(<void*>self.data_ptr)
        pass


cdef class cppilqr:
    cdef ilqr *il

    def __cinit__(self, N=1000):
        self.il = new ilqr(N)

    # def __dealloc__(self):
    #     del self.il

    def set_parameters(self, integrator_ind, delta_t):
        self.il.set_parameters(integrator_ind, delta_t)

    def set_start(self, pos1, pos2, vel1, vel2):
        self.il.set_start(pos1, pos2, vel1, vel2)

    def set_goal(self, pos1, pos2, vel1, vel2):
        self.il.set_goal(pos1, pos2, vel1, vel2)

    def set_cost_parameters(self,
                            double su1, double su2,
                            double sp1, double sp2,
                            double sv1, double sv2,
                            double sen,
                            double fp1, double fp2,
                            double fv1, double fv2,
                            double fen):
        self.il.set_cost_parameters(su1, su2,
                                    sp1, sp2,
                                    sv1, sv2,
                                    sen,
                                    fp1, fp2,
                                    fv1, fv2,
                                    fen)

    def set_model_parameters(self,
                             m1, m2,
                             l1, l2,
                             cm1, cm2,
                             I1, I2,
                             d1, d2,
                             cf1, cf2,
                             g,
                             tl1, tl2):
        self.il.set_model_parameters(m1, m2,
                                     l1, l2,
                                     cm1, cm2,
                                     I1, I2,
                                     d1, d2,
                                     cf1, cf2,
                                     g,
                                     tl1, tl2)

    def set_u_init_traj(self, u1, u2):
        cdef np.ndarray[double, ndim=1, mode="c"] uu1
        cdef np.ndarray[double, ndim=1, mode="c"] uu2
        uu1 = u1
        uu2 = u2
        self.il.set_u_init_traj(&uu1[0], &uu2[0])

    def set_x_init_traj(self, p1, p2, v1, v2):
        cdef np.ndarray[double, ndim=1, mode="c"] pp1
        cdef np.ndarray[double, ndim=1, mode="c"] pp2
        cdef np.ndarray[double, ndim=1, mode="c"] vv1
        cdef np.ndarray[double, ndim=1, mode="c"] vv2
        pp1 = p1
        pp2 = p2
        vv1 = v1
        vv2 = v2
        self.il.set_x_init_traj(&pp1[0], &pp2[0], &vv1[0], &vv2[0])

    def run_ilqr(self, max_iter, break_cost_redu, regu_init, max_regu, min_regu):
        self.il.run_ilqr(max_iter, break_cost_redu, regu_init, max_regu, min_regu)

    def get_u1_traj(self):
        N = self.il.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.il.get_u1_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N-1, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_u2_traj(self):
        N = self.il.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.il.get_u2_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N-1, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_p1_traj(self):
        N = self.il.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.il.get_p1_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_p2_traj(self):
        N = self.il.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.il.get_p2_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_v1_traj(self):
        N = self.il.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.il.get_v1_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_v2_traj(self):
        N = self.il.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.il.get_v2_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def save_trajectory_csv(self):
        self.il.save_trajectory_csv();


cdef class cppilqrmpc:
    cdef ilqr_mpc *ilmpc

    def __cinit__(self, N=1000, N_init=5000):
        self.ilmpc = new ilqr_mpc(N, N_init)

    # def __dealloc__(self):
    #     del self.ilmpc

    def set_parameters(self, integrator_ind, delta_t,
                       max_it, break_cost_red, regu_ini,
                       max_reg, min_reg, shifting):
        self.ilmpc.set_parameters(integrator_ind,
                                  delta_t,
                                  max_it,
                                  break_cost_red,
                                  regu_ini,
                                  max_reg,
                                  min_reg,
                                  shifting)

    def set_start(self, pos1, pos2, vel1, vel2):
        self.ilmpc.set_start(pos1, pos2, vel1, vel2)

    def set_goal(self, pos1, pos2, vel1, vel2):
        self.ilmpc.set_goal(pos1, pos2, vel1, vel2)

    def set_cost_parameters(self,
                            double su1, double su2,
                            double sp1, double sp2,
                            double sv1, double sv2,
                            double sen,
                            double fp1, double fp2,
                            double fv1, double fv2,
                            double fen):
        self.ilmpc.set_cost_parameters(su1, su2,
                                       sp1, sp2,
                                       sv1, sv2,
                                       sen,
                                       fp1, fp2,
                                       fv1, fv2,
                                       fen)


    def set_final_cost_parameters(self,
                                  double su1, double su2,
                                  double sp1, double sp2,
                                  double sv1, double sv2,
                                  double sen,
                                  double fp1, double fp2,
                                  double fv1, double fv2,
                                  double fen):
        self.ilmpc.set_final_cost_parameters(su1, su2,
                                             sp1, sp2,
                                             sv1, sv2,
                                             sen,
                                             fp1, fp2,
                                             fv1, fv2,
                                             fen)

    def set_model_parameters(self,
                             double m1, double m2,
                             double l1, double l2,
                             double cm1, double cm2,
                             double I1, double I2,
                             double d1, double d2,
                             double cf1, double cf2,
                             double g,
                             double tl1, double tl2):
        self.ilmpc.set_model_parameters(m1, m2,
                                        l1, l2,
                                        cm1, cm2,
                                        I1, I2,
                                        d1, d2,
                                        cf1, cf2,
                                        g,
                                        tl1, tl2)

    def set_u_init_traj(self, u1, u2):
        cdef np.ndarray[double, ndim=1, mode="c"] uu1
        cdef np.ndarray[double, ndim=1, mode="c"] uu2
        uu1 = u1
        uu2 = u2
        self.ilmpc.set_u_init_traj(&uu1[0], &uu2[0])

    def set_x_init_traj(self, p1, p2, v1, v2, traj_stab):
        cdef np.ndarray[double, ndim=1, mode="c"] pp1
        cdef np.ndarray[double, ndim=1, mode="c"] pp2
        cdef np.ndarray[double, ndim=1, mode="c"] vv1
        cdef np.ndarray[double, ndim=1, mode="c"] vv2
        pp1 = p1
        pp2 = p2
        vv1 = v1
        vv2 = v2
        self.ilmpc.set_x_init_traj(&pp1[0], &pp2[0], &vv1[0], &vv2[0], traj_stab)

    def get_control_output(self, p1, p2, v1, v2):
        u = self.ilmpc.get_control_output(p1, p2, v1, v2)
        return u

    def get_u1_traj(self):
        N = self.ilmpc.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.ilmpc.get_u1_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N-1, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_u2_traj(self):
        N = self.ilmpc.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.ilmpc.get_u2_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N-1, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_p1_traj(self):
        N = self.ilmpc.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.ilmpc.get_p1_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_p2_traj(self):
        N = self.ilmpc.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.ilmpc.get_p2_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_v1_traj(self):
        N = self.ilmpc.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.ilmpc.get_v1_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

    def get_v2_traj(self):
        N = self.ilmpc.get_N()
        cdef double *vec
        cdef np.ndarray ar
        vec = self.ilmpc.get_v2_traj()

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(N, <void*> vec)
        ar = np.array(array_wrapper, copy=False)
        ar.base = <PyObject*> array_wrapper
        Py_INCREF(array_wrapper)
        return ar

