#ifndef ILQR_HPP
#define ILQR_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "../../../model/src/dp_plant.hpp"
#include "../../../simulator/src/simulator.hpp"

class ilqr {

public:
  static const int n_x = 4;
  static const int n_u = 1;

private:
  double mass1, mass2;
  double length1, length2;
  double com1, com2;
  double inertia1, inertia2;
  double damping1, damping2;
  double coulomb_friction1, coulomb_friction2;
  double gravity;
  double torque_limit1, torque_limit2;

  DPPlant plant;
  Simulator sim;
  int active_act = 1;

  std::string integrator;
  double dt;
  Eigen::Vector<double, n_x> discrete_dynamics(Eigen::Vector<double, n_x>,
                                               Eigen::Vector<double, n_u>);

  Eigen::Vector<double, n_x> x0, goal;
  double goal_energy;

  double sCu1, sCu2, sCp1, sCp2, sCv1, sCv2, sCen, fCp1, fCp2, fCv1, fCv2, fCen;
  double stage_cost(Eigen::Vector<double, n_x>, Eigen::Vector<double, n_u>,
                    int);
  double final_cost(Eigen::Vector<double, n_x>);
  double calculate_cost(bool);

  Eigen::Matrix<double, n_x, n_x> dyn_x;    // (4,4)
  Eigen::Matrix<double, n_x, n_u> dyn_u;    // (4,1)
  Eigen::Vector<double, n_x> stage_x;       // (4)
  Eigen::Vector<double, n_u> stage_u;       // (1)
  Eigen::Matrix<double, n_x, n_x> stage_xx; // (4,4)
  Eigen::Matrix<double, n_u, n_x> stage_ux; //(1,2)
  Eigen::Matrix<double, n_u, n_u> stage_uu; // (1,2)
  Eigen::Vector<double, n_x> final_x;       // (4)
  Eigen::Matrix<double, n_x, n_x> final_xx; // (4,4)

  void compute_dynamics_x(Eigen::Vector<double, n_x>,
                          Eigen::Vector<double, n_u>);
  void compute_dynamics_u(Eigen::Vector<double, n_x>,
                          Eigen::Vector<double, n_u>);
  void compute_stage_x(Eigen::Vector<double, n_x>, Eigen::Vector<double, n_u>,
                       int);
  void compute_stage_u(Eigen::Vector<double, n_x>, Eigen::Vector<double, n_u>,
                       int);
  void compute_stage_xx(Eigen::Vector<double, n_x>, Eigen::Vector<double, n_u>,
                        int);
  void compute_stage_ux(Eigen::Vector<double, n_x>, Eigen::Vector<double, n_u>,
                        int);
  void compute_stage_uu(Eigen::Vector<double, n_x>, Eigen::Vector<double, n_u>,
                        int);
  void compute_final_x(Eigen::Vector<double, n_x>);
  void compute_final_xx(Eigen::Vector<double, n_x>);
  void compute_derivatives(Eigen::Vector<double, n_x>,
                           Eigen::Vector<double, n_u>, int);

  void rollout();
  void calculate_Q_terms();
  void calculate_V_terms();
  void calculate_gains(double);
  double expected_cost_reduction();
  void forward_pass();
  double backward_pass(double);

  Eigen::Vector<double, n_u> Q_u, k;
  Eigen::Matrix<double, n_u, n_u> Q_uu, Q_uu_regu;
  Eigen::Matrix<double, n_u, n_x> Q_ux, K;
  Eigen::Vector<double, n_x> V_x, Q_x;
  Eigen::Matrix<double, n_x, n_x> Q_xx, V_xx;

  // helper variables
  double en_diff;
  // double eps = 1e-6;
  int verbose = 0;

  int N;
  Eigen::Vector<double, n_u> *k_traj = new Eigen::Vector<double, n_u>[N];
  Eigen::Matrix<double, n_u, n_x> *K_traj =
      new Eigen::Matrix<double, n_u, n_x>[N];

  ////double regu;
  //// make n dimensional (for now as double for python bindings)
  double *u1_traj_doubles = new double[N - 1];
  double *u2_traj_doubles = new double[N - 1];
  double *p1_traj_doubles = new double[N];
  double *p2_traj_doubles = new double[N];
  double *v1_traj_doubles = new double[N];
  double *v2_traj_doubles = new double[N];

public:
  ilqr();
  ilqr(int);
  ~ilqr();

  void set_verbose(int);
  void read_parameter_file(std::string);
  void set_parameters(int, double);
  void set_model_parameters(double, double, double, double, double, double,
                            double, double, double, double, double, double,
                            double, double, double);
  void set_cost_parameters(double, double, double, double, double, double,
                           double, double, double, double, double, double);
  void set_start(Eigen::Vector<double, n_x>);
  void set_start(double, double, double, double);
  void set_goal(Eigen::Vector<double, n_x>);
  void set_goal(double, double, double, double);
  void load_goal_traj(std::string);
  void set_goal_traj(double p1[], double p2[], double v1[], double v2[],
                     int from, int to);
  void set_goal_traj(Eigen::Vector<double, n_x> x_tr[], int from, int to);
  void set_goal_traj(Eigen::Vector<double, n_x> x_tr[],
                     Eigen::Vector<double, n_u> u_tr[], int from, int to);
  void set_u_init_traj(double u1[], double u2[]);
  void set_u_init_traj(Eigen::Vector<double, n_u> utrj[]);
  void set_x_init_traj(double p1[], double p2[], double v1[], double v2[]);
  void set_x_init_traj(Eigen::Vector<double, n_x> xtrj[]);

  void run_ilqr(int, double, double, double, double);

  Eigen::Vector<double, n_u> *u_traj = new Eigen::Vector<double, n_u>[N - 1];
  Eigen::Vector<double, n_u> *u_traj_new =
      new Eigen::Vector<double, n_u>[N - 1];
  Eigen::Vector<double, n_x> *x_traj = new Eigen::Vector<double, n_x>[N];
  Eigen::Vector<double, n_x> *x_traj_new = new Eigen::Vector<double, n_x>[N];

  Eigen::Vector<double, n_x> *goal_traj_x = new Eigen::Vector<double, n_x>[N];
  Eigen::Vector<double, n_u> *goal_traj_u =
      new Eigen::Vector<double, n_u>[N - 1];
  double *goal_traj_energy = new double[N];

  bool warm_start_x = false;
  bool warm_start_u = false;

  int get_N();
  // int get_n_x() {return n_x;};
  // int get_n_u() {return n_u;};
  double *get_u1_traj();
  double *get_u2_traj();
  double *get_p1_traj();
  double *get_p2_traj();
  double *get_v1_traj();
  double *get_v2_traj();

  Eigen::Vector<double, n_u> *get_u_traj();
  Eigen::Vector<double, n_x> *get_x_traj();

  Eigen::Vector<double, n_u> *best_k_traj = new Eigen::Vector<double, n_u>[N];
  Eigen::Matrix<double, n_u, n_x> *best_K_traj =
      new Eigen::Matrix<double, n_u, n_x>[N];

  void save_trajectory_csv();
  void save_trajectory_csv(std::string);
};

#endif // ILQR_HPP
