# distutils: language = c++
# cython: language_level=3

#cimport cython
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "../model/src/dp_plant.cpp":
    pass

cdef extern from "../simulator/src/simulator.cpp":
    pass

cdef extern from "../controllers/ilqr/src/ilqr.hpp":
    cdef cppclass ilqr:
        ilqr() except +
        ilqr(int N) except +
        void set_parameters(int integrator_ind, double delta_t)
        void set_start(double pos1, double pos2,
                       double vel1, double vel2)
        void set_goal(double pos1, double pos2,
                      double vel1, double vel2)
        void set_cost_parameters(double su1, double su2,
                                 double sp1, double sp2,
                                 double sv1, double sv2,
                                 double sen,
                                 double fp1, double fp2,
                                 double fv1, double fv2,
                                 double fen)
        void set_model_parameters(double m1, double m2,
                                  double l1, double l2,
                                  double cm1, double cm1,
                                  double I1, double I2,
                                  double d1, double d2,
                                  double cf1, double cf2,
                                  double g,
                                  double tl1, double tl2)
        void set_u_init_traj(double u1[], double u2[])
        void set_x_init_traj(double p1[], double p2[],
                             double v1[], double v2[])
        void run_ilqr(int max_iter, double break_cost_redu, double regu_init,
                      double max_regu, double min_regu)
        #vector[double] * get_u_traj()
        double *get_u1_traj()
        double *get_u2_traj()
        double *get_p1_traj()
        double *get_p2_traj()
        double *get_v1_traj()
        double *get_v2_traj()
        int get_N()
        void save_trajectory_csv()
        const int N

cdef extern from "../controllers/ilqr/src/ilqr_mpc.hpp":
    cdef cppclass ilqr_mpc:
        ilqr_mpc() except +
        ilqr_mpc(int N, int N_init) except +
        void set_parameters(int integrator_ind, double delta_t, int max_it,
                            double break_cost_red, double regu_ini,
                            double max_reg, double min_reg)
        void set_start(double pos1, double pos2,
                       double vel1, double vel2)
        void set_goal(double pos1, double pos2,
                      double vel1, double vel2)
        void set_cost_parameters(double su1, double su2,
                                 double sp1, double sp2,
                                 double sv1, double sv2,
                                 double sen,
                                 double fp1, double fp2,
                                 double fv1, double fv2,
                                 double fen)
        void set_model_parameters(double m1, double m2,
                                  double l1, double l2,
                                  double cm1, double cm1,
                                  double I1, double I2,
                                  double d1, double d2,
                                  double cf1, double cf2,
                                  double g,
                                  double tl1, double tl2)
        void set_u_init_traj(double u1[], double u2[])
        void set_x_init_traj(double p1[], double p2[],
                             double v1[], double v2[],
                             bool traj_stab)
        double get_control_output(double p1, double p2,
                                  double v1, double v2)
        double *get_u1_traj()
        double *get_u2_traj()
        double *get_p1_traj()
        double *get_p2_traj()
        double *get_v1_traj()
        double *get_v2_traj()

        int get_N()
        const int N
        const int N_init
