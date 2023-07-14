#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "ilqr.hpp"
#include "ilqr_mpc.hpp"

ilqr_mpc::ilqr_mpc() : N(1000), N_init(5000) { ilqr_calc = new ilqr(N); }

ilqr_mpc::ilqr_mpc(int n, int n_init) : N(n), N_init(n_init) {
  ilqr_calc = new ilqr(N);
}

ilqr_mpc::~ilqr_mpc() {
  delete[] u1_mpctraj_doubles;
  delete[] u2_mpctraj_doubles;
  delete[] p1_mpctraj_doubles;
  delete[] p2_mpctraj_doubles;
  delete[] v1_mpctraj_doubles;
  delete[] v2_mpctraj_doubles;
  delete[] u_traj;
  delete[] x_traj;
  delete[] u_init_traj;
  delete[] x_init_traj;
  delete ilqr_calc;
}

void ilqr_mpc::set_verbose(int ver, int ver_il) {
  verbose = ver;
  verbose_il = ver_il;
  ilqr_calc->set_verbose(verbose_il);
}

int ilqr_mpc::get_N() { return N; }

void ilqr_mpc::set_parameters(int integ_ind, double delta_t, int max_it,
                              double break_cost_red, double regu_ini,
                              double max_reg, double min_reg, int shifting) {
  integrator_ind = integ_ind;
  if (integ_ind == 0) {
    integrator = "euler";
  } else {
    integrator = "runge_kutta";
  }
  dt = delta_t;
  ilqr_calc->set_parameters(integrator_ind, dt);
  max_iter = max_it;
  break_cost_redu = break_cost_red;
  regu_init = regu_ini;
  max_regu = max_reg;
  min_regu = min_reg;
  shift_every = shifting;
}

void ilqr_mpc::read_parameter_file(std::string configfile) {

  YAML::Node config = YAML::LoadFile(configfile);
  if (config["mass1"]) {
    mass1 = config["mass1"].as<double>();
  }
  if (config["mass2"]) {
    mass2 = config["mass2"].as<double>();
  }
  if (config["length1"]) {
    length1 = config["length1"].as<double>();
  }
  if (config["length2"]) {
    length2 = config["length2"].as<double>();
  }
  if (config["com1"]) {
    com1 = config["com1"].as<double>();
  }
  if (config["com2"]) {
    com2 = config["com2"].as<double>();
  }
  if (config["damping1"]) {
    damping1 = config["damping1"].as<double>();
  }
  if (config["damping2"]) {
    damping2 = config["damping2"].as<double>();
  }
  if (config["coulomb_friction1"]) {
    coulomb_friction1 = config["coulomb_friction1"].as<double>();
  }
  if (config["coulomb_friction2"]) {
    coulomb_friction2 = config["coulomb_friction2"].as<double>();
  }
  if (config["gravity"]) {
    gravity = config["gravity"].as<double>();
  }
  if (config["inertia1"]) {
    inertia1 = config["inertia1"].as<double>();
  } else {
    inertia1 = mass1 * com1 * com1;
  }
  if (config["inertia2"]) {
    inertia2 = config["inertia2"].as<double>();
  } else {
    inertia2 = mass2 * com2 * com2;
  }
  if (config["torque_limit1"]) {
    torque_limit1 = config["torque_limit1"].as<double>();
  }
  if (config["torque_limit2"]) {
    torque_limit2 = config["torque_limit2"].as<double>();
  }
  if (config["deltaT"]) {
    dt = config["deltaT"].as<double>();
  }
  if (config["integrator"]) {
    integrator_ind = config["integrator"].as<int>();
    if (integrator_ind == 0) {
      integrator = "euler";
    } else {
      integrator = "runge_kutta";
    }
  }
  if (config["start_pos1"]) {
    x0(0) = config["start_pos1"].as<double>();
  }
  if (config["start_pos2"]) {
    x0(1) = config["start_pos2"].as<double>();
  }
  if (config["start_vel1"]) {
    x0(2) = config["start_vel1"].as<double>();
  }
  if (config["start_vel2"]) {
    x0(3) = config["start_vel2"].as<double>();
  }
  // if (config["goal_pos1"]) {goal(0)=config["goal_pos1"].as<double>();}
  // if (config["goal_pos2"]) {goal(1)=config["goal_pos2"].as<double>();}
  if (config["goal_pos1"]) {
    goal(0) = std::fmod(config["goal_pos1"].as<double>(), 2. * M_PI);
  }
  if (config["goal_pos2"]) {
    goal(1) =
        std::fmod(config["goal_pos2"].as<double>() + M_PI, 2. * M_PI) - M_PI;
  }
  if (config["goal_vel1"]) {
    goal(2) = config["goal_vel1"].as<double>();
  }
  if (config["goal_vel2"]) {
    goal(3) = config["goal_vel2"].as<double>();
  }
  if (config["sCu1"]) {
    sCu1 = config["sCu1"].as<double>();
  }
  if (config["sCu2"]) {
    sCu2 = config["sCu2"].as<double>();
  }
  if (config["sCp1"]) {
    sCp1 = config["sCp1"].as<double>();
  }
  if (config["sCp2"]) {
    sCp2 = config["sCp2"].as<double>();
  }
  if (config["sCv1"]) {
    sCv1 = config["sCv1"].as<double>();
  }
  if (config["sCv2"]) {
    sCv2 = config["sCv2"].as<double>();
  }
  if (config["sCen"]) {
    sCen = config["sCen"].as<double>();
  }
  if (config["fCp1"]) {
    fCp1 = config["fCp1"].as<double>();
  }
  if (config["fCp2"]) {
    fCp2 = config["fCp2"].as<double>();
  }
  if (config["fCv1"]) {
    fCv1 = config["fCv1"].as<double>();
  }
  if (config["fCv2"]) {
    fCv2 = config["fCv2"].as<double>();
  }
  if (config["fCen"]) {
    fCen = config["fCen"].as<double>();
  }
  if (config["N"]) {
    N = config["N"].as<int>();
  }

  if (config["verbose"]) {
    verbose = config["verbose"].as<int>();
  }
  if (config["verbose_il"]) {
    verbose_il = config["verbose_il"].as<int>();
  }
  if (config["max_iter"]) {
    max_iter = config["max_iter"].as<int>();
  }
  if (config["break_cost_redu"]) {
    break_cost_redu = config["break_cost_redu"].as<double>();
  }
  if (config["regu_init"]) {
    regu_init = config["regu_init"].as<double>();
  }
  if (config["max_regu"]) {
    max_regu = config["max_regu"].as<double>();
  }
  if (config["min_regu"]) {
    min_regu = config["min_regu"].as<double>();
  }

  ilqr_calc->set_verbose(verbose_il);
  ilqr_calc->set_parameters(integrator_ind, dt);
  ilqr_calc->set_cost_parameters(sCu1, sCu2, sCp1, sCp2, sCv1, sCv2, sCen, fCp1,
                                 fCp2, fCv2, fCv2, fCen);
  ilqr_calc->set_model_parameters(mass1, mass2, length1, length2, com1, com2,
                                  inertia1, inertia2, damping1, damping2,
                                  coulomb_friction1, coulomb_friction2, gravity,
                                  torque_limit1, torque_limit2);
  if (torque_limit1 > 0.) {
    active_act = 0;
  } else if (torque_limit2 > 0.) {
    active_act = 1;
  }
}
void ilqr_mpc::set_cost_parameters(double su1, double su2, double sp1,
                                   double sp2, double sv1, double sv2,
                                   double sen, double fp1, double fp2,
                                   double fv1, double fv2, double fen) {
  sCu1 = su1;
  sCu2 = su2;
  sCp1 = sp1;
  sCp2 = sp2;
  sCv1 = sv1;
  sCv2 = sv2;
  sCen = sen;
  fCp1 = fp1;
  fCp2 = fp2;
  fCv1 = fv1;
  fCv2 = fv2;
  fCen = fen;

  ilqr_calc->set_cost_parameters(su1, su2, sp1, sp2, sv1, sv2, sen, fp1, fp2,
                                 fv2, fv2, fen);

  // set final parameter defaults to the same values
  f_sCu1 = su1;
  f_sCu2 = su2;
  f_sCp1 = sp1;
  f_sCp2 = sp2;
  f_sCv1 = sv1;
  f_sCv2 = sv2;
  f_sCen = sen;
  f_fCp1 = fp1;
  f_fCp2 = fp2;
  f_fCv1 = fv1;
  f_fCv2 = fv2;
  f_fCen = fen;
}

void ilqr_mpc::set_final_cost_parameters(double su1, double su2, double sp1,
                                         double sp2, double sv1, double sv2,
                                         double sen, double fp1, double fp2,
                                         double fv1, double fv2, double fen) {
  f_sCu1 = su1;
  f_sCu2 = su2;
  f_sCp1 = sp1;
  f_sCp2 = sp2;
  f_sCv1 = sv1;
  f_sCv2 = sv2;
  f_sCen = sen;
  f_fCp1 = fp1;
  f_fCp2 = fp2;
  f_fCv1 = fv1;
  f_fCv2 = fv2;
  f_fCen = fen;
}

void ilqr_mpc::set_model_parameters(double m1, double m2, double l1, double l2,
                                    double cm1, double cm2, double I1,
                                    double I2, double d1, double d2, double cf1,
                                    double cf2, double g, double tl1,
                                    double tl2) {
  mass1 = m1;
  mass2 = m2;
  length1 = l1;
  length2 = l2;
  com1 = cm1;
  com2 = cm2;
  inertia1 = I1;
  inertia2 = I2;
  damping1 = d1;
  damping2 = d2;
  coulomb_friction1 = cf1;
  coulomb_friction2 = cf2;
  gravity = g;
  torque_limit1 = tl1;
  torque_limit2 = tl2;

  ilqr_calc->set_model_parameters(m1, m2, l1, l2, cm1, cm2, I1, I2, d1, d2, cf1,
                                  cf2, g, tl1, tl2);
  if (torque_limit1 > 0.) {
    active_act = 0;
  } else if (torque_limit2 > 0.) {
    active_act = 1;
  }
}

void ilqr_mpc::set_start(Eigen::Vector<double, ilqr::n_x> x) {
  x0 = x;
  ilqr_calc->set_start(x0);
}

void ilqr_mpc::set_start(double pos1, double pos2, double vel1, double vel2) {
  x0(0) = pos1;
  x0(1) = pos2;
  x0(2) = vel1;
  x0(3) = vel2;
  ilqr_calc->set_start(x0);
}

void ilqr_mpc::set_goal(Eigen::Vector<double, ilqr::n_x> x) {
  goal = x;
  goal(0) = std::fmod(goal(0), 2. * M_PI);
  goal(1) = std::fmod(goal(1) + M_PI, 2. * M_PI) - M_PI;
  ilqr_calc->set_goal(goal);
}

void ilqr_mpc::set_goal(double pos1, double pos2, double vel1, double vel2) {
  goal(0) = std::fmod(pos1, 2. * M_PI);
  goal(1) = std::fmod(pos2 + M_PI, 2. * M_PI) - M_PI;
  goal(2) = vel1;
  goal(3) = vel2;
  ilqr_calc->set_goal(goal);
}

void ilqr_mpc::set_u_init_traj(double u1[], double u2[]) {
  int Nmin = std::min(N_init, N);
  if (active_act == 0) {
    for (int i = 0; i < N_init - 1; i++) {
      u_init_traj[i](0) = u1[i];
      // u_init_traj[i](1) = u2[i];
    }
  } else {
    for (int i = 0; i < N_init - 1; i++) {
      u_init_traj[i](0) = u2[i];
      // u_init_traj[i](1) = u2[i];
    }
  }
  for (int i = 0; i < Nmin - 1; i++) {
    u_traj[i] = u_init_traj[i];
  }

  // ilqr_calc->set_u_init_traj(u_traj);
}

void ilqr_mpc::set_u_init_traj(Eigen::Vector<double, ilqr::n_u> utrj[]) {
  for (int i = 0; i < N_init - 1; i++) {
    u_init_traj[i] = utrj[i];
  }
  int Nmin = std::min(N_init, N);
  for (int i = 0; i < Nmin - 1; i++) {
    u_traj[i] = u_init_traj[i];
  }
  // ilqr_calc->set_u_init_traj(u_traj);
}

void ilqr_mpc::set_x_init_traj(double p1[], double p2[], double v1[],
                               double v2[], bool traj_stab) {
  int Nmin = std::min(N_init, N);
  for (int i = 0; i < N_init; i++) {
    x_init_traj[i](0) = p1[i];
    x_init_traj[i](1) = p2[i];
    x_init_traj[i](2) = v1[i];
    x_init_traj[i](3) = v2[i];
  }
  for (int i = 0; i < Nmin; i++) {
    x_traj[i] = x_init_traj[i];
  }
  // ilqr_calc->set_x_init_traj(x_traj);
  trajectory_stabilization = traj_stab;
}

void ilqr_mpc::set_x_init_traj(Eigen::Vector<double, ilqr::n_x> xtrj[],
                               bool traj_stab) {
  // x_init_traj = xtrj;
  for (int i = 0; i < N_init; i++) {
    x_init_traj[i] = xtrj[i];
  }
  int Nmin = std::min(N_init, N);
  for (int i = 0; i < Nmin; i++) {
    x_traj[i] = x_init_traj[i];
  }
  // ilqr_calc->set_x_init_traj(x_traj);
  trajectory_stabilization = traj_stab;
}

void ilqr_mpc::shift_trajs(int s) {
  for (int i = 0; i < N - 2; i++) {
    u_traj[i](0) = u_traj[i + 1](0);
  }
  for (int i = 0; i < N - 1; i++) {
    x_traj[i](0) = x_traj[i + 1](0);
    x_traj[i](1) = x_traj[i + 1](1);
    x_traj[i](2) = x_traj[i + 1](2);
    x_traj[i](3) = x_traj[i + 1](3);
  }

  if (N + s < N_init) {
    u_traj[N - 2](0) = u_init_traj[N + s - 1](0);
    x_traj[N - 1](0) = x_init_traj[N + s](0);
    x_traj[N - 1](1) = x_init_traj[N + s](1);
    x_traj[N - 1](2) = x_init_traj[N + s](2);
    x_traj[N - 1](3) = x_init_traj[N + s](3);
  } else {
    u_traj[N - 2](0) = 0.0; // todo: acro/pendu
  }
}

// Eigen::Vector<double, ilqr::n_u>
// ilqr_mpc::get_control_output(Eigen::Vector<double, ilqr::n_x> x){
double ilqr_mpc::get_control_output(Eigen::Vector<double, ilqr::n_x> x) {

  // Eigen::Vector<double, ilqr::n_x> y = x;
  // y(0) = std::fmod(y(0), 2.*M_PI);
  // y(1) = std::fmod(y(1)+M_PI, 2.*M_PI) - M_PI;

  // std::cout << "MPC get output: integrator_ind " << integrator_ind <<
  // std::endl; std::cout << "MPC get output: max_iter " << max_iter <<
  // std::endl; ilqr_calc->set_parameters(integrator_ind, dt);
  // ilqr_calc->set_model_parameters(mass1, mass2,
  //                                 length1, length2,
  //                                 com1, com2,
  //                                 inertia1, inertia2,
  //                                 damping1, damping2,
  //                                 coulomb_friction1, coulomb_friction2,
  //                                 gravity,
  //                                 torque_limit1, torque_limit2);
  // ilqr_calc->set_cost_parameters(sCu1, sCu2,
  //                                sCp1, sCp2,
  //                                sCv1, sCv2,
  //                                sCen,
  //                                fCp1, fCp2,
  //                                fCv2, fCv2,
  //                                fCen);

  // set the running goal to N steps in the future on the precomputed traj
  Eigen::Vector<double, ilqr::n_x> x0, running_goal;
  int N_hor = std::min(counter + N, N_init - 1);
  int counter_cap = std::min(counter, N_init - 1);
  // int N_hor_cap = std::min(counter+N, N-1);
  running_goal(0) = x_init_traj[N_hor](0);
  running_goal(1) = x_init_traj[N_hor](1);
  running_goal(2) = x_init_traj[N_hor](2);
  running_goal(3) = x_init_traj[N_hor](3);

  ilqr_calc->set_start(x);
  ilqr_calc->set_u_init_traj(u_traj);
  // ilqr_calc->set_x_init_traj(x_traj);

  if ((counter + N < N_init - 1) and trajectory_stabilization) {
    ilqr_calc->set_goal(running_goal);
    // ilqr_calc->set_goal_traj(x_init_traj, counter_cap, N_hor_cap);
    ilqr_calc->set_goal_traj(x_init_traj, u_init_traj, counter_cap, N_hor);
  } else {
    ilqr_calc->set_cost_parameters(f_sCu1, f_sCu2, f_sCp1, f_sCp2, f_sCv1,
                                   f_sCv2, f_sCen, f_fCp1, f_fCp2, f_fCv2,
                                   f_fCv2, f_fCen);
    ilqr_calc->set_goal(goal);
  }
  ilqr_calc->run_ilqr(max_iter, break_cost_redu, regu_init, max_regu, min_regu);

  // u_traj = ilqr_calc->get_u_traj();
  // x_traj = ilqr_calc->get_x_traj();

  for (int i = 0; i < N - 1; i++) {
    u_traj[i] = ilqr_calc->u_traj[i];
  }
  for (int i = 0; i < N; i++) {
    x_traj[i] = ilqr_calc->x_traj[i];
  }

  // Eigen::Vector<double, ilqr::n_u> u;
  // Eigen::Vector<double, 2> u_full;

  double u;

  // u[0] = 0.;
  // u[0] = u_traj[0](0);
  if (trajectory_stabilization) {
    u = u_traj[0](0);
  } else {
    u = u_traj[0](0) +
        ilqr_calc->best_K_traj[0](0, 0) * (x_init_traj[counter](0) - x(0)) +
        ilqr_calc->best_K_traj[0](0, 1) * (x_init_traj[counter](1) - x(1)) +
        ilqr_calc->best_K_traj[0](0, 2) * (x_init_traj[counter](2) - x(2)) +
        ilqr_calc->best_K_traj[0](0, 3) * (x_init_traj[counter](3) - x(3));
  }

  if ((counter + 1) % shift_every == 0) {
    shift_trajs(counter);
  }
  counter += 1;

  if (verbose > 0) {
    std::cout << "Step " << counter << " ";
    std::cout << x(0) << ", ";
    std::cout << x(1) << ", ";
    std::cout << x(2) << ", ";
    std::cout << x(3) << ", ";
    std::cout << 0.0 << ", ";
    std::cout << u;
    if (counter < N_init) {
      std::cout << ", (" << u_init_traj[counter][0] << ")";
      if (pow(pow(u - u_init_traj[counter][0], 2.0), 0.5) > 0.01) {
        std::cout << " <-------------";
      }
    }
    std::cout << std::endl;
  }

  return u;
}

double ilqr_mpc::get_control_output(double p1, double p2, double v1,
                                    double v2) {
  Eigen::Vector<double, ilqr::n_x> x;
  x(0) = p1;
  x(1) = p2;
  x(2) = v1;
  x(3) = v2;
  double u = get_control_output(x);
  return u;
}

double *ilqr_mpc::get_u1_traj() {
  if (active_act == 0) {
    for (int i = 0; i < N - 1; i++) {
      u1_mpctraj_doubles[i] = ilqr_calc->u_traj[i](0);
    }
  } else {
    for (int i = 0; i < N - 1; i++) {
      u1_mpctraj_doubles[i] = 0.0;
    }
  }
  return u1_mpctraj_doubles;
}

double *ilqr_mpc::get_u2_traj() {
  if (active_act == 1) {
    for (int i = 0; i < N - 1; i++) {
      u2_mpctraj_doubles[i] = ilqr_calc->u_traj[i](0);
    }
  } else {
    for (int i = 0; i < N - 1; i++) {
      u2_mpctraj_doubles[i] = 0.0;
    }
  }
  return u2_mpctraj_doubles;
}

double *ilqr_mpc::get_p1_traj() {
  for (int i = 0; i < N; i++) {
    p1_mpctraj_doubles[i] = ilqr_calc->x_traj[i](0);
  }
  return p1_mpctraj_doubles;
}

double *ilqr_mpc::get_p2_traj() {
  for (int i = 0; i < N; i++) {
    p2_mpctraj_doubles[i] = ilqr_calc->x_traj[i](1);
  }
  return p2_mpctraj_doubles;
}

double *ilqr_mpc::get_v1_traj() {
  for (int i = 0; i < N; i++) {
    v1_mpctraj_doubles[i] = ilqr_calc->x_traj[i](2);
  }
  return v1_mpctraj_doubles;
}

double *ilqr_mpc::get_v2_traj() {
  for (int i = 0; i < N; i++) {
    v2_mpctraj_doubles[i] = ilqr_calc->x_traj[i](3);
  }
  return v2_mpctraj_doubles;
}
