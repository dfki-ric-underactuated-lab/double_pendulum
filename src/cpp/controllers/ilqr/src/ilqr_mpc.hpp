#ifndef ILQR_MPC_HPP
#define ILQR_MPC_HPP

#include <string>
#include <Eigen/Dense>
#include <vector>

#include "ilqr.hpp"

class ilqr_mpc{

public:
    ilqr_mpc();
    ilqr_mpc(int, int);
    ~ilqr_mpc();

    int N;
    int N_init;
    ilqr* ilqr_calc;

private: 

    double mass1, mass2;
    double length1, length2;
    double com1, com2;
    double inertia1, inertia2;
    double damping1, damping2;
    double coulomb_friction1, coulomb_friction2;
    double gravity;
    double torque_limit1, torque_limit2;

    double sCu1, sCu2, sCp1, sCp2, sCv1, sCv2, sCen, fCp1, fCp2, fCv1, fCv2, fCen;

    Eigen::Vector<double, ilqr::n_x> x0, goal;

    std::string integrator;
    int integrator_ind;
    double dt;

    int max_iter;
    double break_cost_redu;
    double regu_init, max_regu, min_regu;

    int counter = 0;

    int verbose = 0;

public:

    void set_parameters(int, double, int, double, double,
                        double, double);
    void read_parameter_file(std::string);
    void set_model_parameters(double, double, double, double, double,
                              double, double, double, double, double,
                              double, double, double, double, double);
    void set_cost_parameters(double, double, double, double, double,
                             double, double, double, double, double,
                             double, double);

    void set_start(Eigen::Vector<double, ilqr::n_x>);
    void set_start(double, double, double, double);
    void set_goal(Eigen::Vector<double, ilqr::n_x>);
    void set_goal(double, double, double, double);
    void set_u_init_traj(double u1[], double u2[]);
    void set_u_init_traj(Eigen::Vector<double, ilqr::n_u> utrj[]);
    void set_x_init_traj(double p1[], double p2[], double v1[], double v2[]);
    void set_x_init_traj(Eigen::Vector<double, ilqr::n_x> xtrj[]);
    void shift_trajs(int);

    Eigen::Vector<double, ilqr::n_u>* u_traj = new Eigen::Vector<double, ilqr::n_u>[N-1];
    Eigen::Vector<double, ilqr::n_x>* x_traj = new Eigen::Vector<double, ilqr::n_x>[N];
    Eigen::Vector<double, ilqr::n_u>* u_init_traj = new Eigen::Vector<double, ilqr::n_u>[N_init-1];
    Eigen::Vector<double, ilqr::n_x>* x_init_traj = new Eigen::Vector<double, ilqr::n_x>[N_init];

    //Eigen::Vector<double, ilqr::n_u> get_control_output(Eigen::Vector<double, ilqr::n_x>);
    double get_control_output(Eigen::Vector<double, ilqr::n_x>);
    double get_control_output(double, double, double, double);
};

#endif // ILQR_MPC_HPP

