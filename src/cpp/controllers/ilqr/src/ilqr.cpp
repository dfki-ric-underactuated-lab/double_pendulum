#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "ilqr.hpp"
// #include "../../../utils/src/csv_reader.hpp"
#include "../../../model/src/dp_plant.hpp"
#include "../../../simulator/src/simulator.hpp"

const int n_x = ilqr::n_x;
const int n_u = ilqr::n_u;

ilqr::ilqr() : N(1000) { srand(time(NULL)); }

ilqr::ilqr(int n) : N(n) { srand(time(NULL)); }

ilqr::~ilqr() {
  delete[] u1_traj_doubles;
  delete[] u2_traj_doubles;
  delete[] p1_traj_doubles;
  delete[] p2_traj_doubles;
  delete[] v1_traj_doubles;
  delete[] v2_traj_doubles;
  delete[] u_traj;
  delete[] x_traj;
  delete[] u_traj_new;
  delete[] x_traj_new;
  delete[] k_traj;
  delete[] K_traj;
  delete[] best_k_traj;
  delete[] best_K_traj;
  delete[] goal_traj_x;
  delete[] goal_traj_u;
  delete[] goal_traj_energy;
}

int ilqr::get_N() { return N; }

void ilqr::set_verbose(int ver) { verbose = ver; }

void ilqr::read_parameter_file(std::string configfile) {

  int integrator_ind;

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
}

void ilqr::set_parameters(int integrator_ind, double delta_t) {
  // n_x = nx;
  // n_u = nu;
  if (integrator_ind == 0) {
    integrator = "euler";
  } else {
    integrator = "runge_kutta";
  }
  dt = delta_t;
}

void ilqr::set_cost_parameters(double su1, double su2, double sp1, double sp2,
                               double sv1, double sv2, double sen, double fp1,
                               double fp2, double fv1, double fv2, double fen) {
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
}

void ilqr::set_model_parameters(double m1, double m2, double l1, double l2,
                                double cm1, double cm2, double I1, double I2,
                                double d1, double d2, double cf1, double cf2,
                                double g, double tl1, double tl2) {
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

  if (torque_limit1 > 0.) {
    active_act = 0;
  } else if (torque_limit2 > 0.) {
    active_act = 1;
  }
  plant = DPPlant(bool(1 - active_act), bool(active_act));
  plant.set_parameters(mass1, mass2, length1, length2, com1, com2, inertia1,
                       inertia2, damping1, damping2, coulomb_friction1,
                       coulomb_friction2, gravity, torque_limit1,
                       torque_limit2);
  sim = Simulator();
  sim.set_plant(plant);
}

void ilqr::set_start(Eigen::Vector<double, n_x> x) { x0 = x; }

void ilqr::set_start(double pos1, double pos2, double vel1, double vel2) {
  x0(0) = pos1;
  x0(1) = pos2;
  x0(2) = vel1;
  x0(3) = vel2;
}

void ilqr::set_goal(Eigen::Vector<double, n_x> x) {
  goal = x;
  goal(0) = std::fmod(goal(0), 2. * M_PI);
  goal(1) = std::fmod(goal(1) + M_PI, 2. * M_PI) - M_PI;
  goal_energy = plant.calculate_total_energy(goal);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < n_x; j++) {
      goal_traj_x[i](j) = goal(j);
    }
    // goal_traj_energy[i] = plant.calculate_total_energy(goal_traj_x[i]);
    goal_traj_energy[i] =
        goal_energy; // save computing time (do not use energy for now)
  }
  for (int i = 0; i < N - 1; i++) {
    for (int j = 0; j < n_u; j++) {
      goal_traj_u[i](j) = 0.; // set default u desired to 0
    }
  }
}

void ilqr::set_goal(double pos1, double pos2, double vel1, double vel2) {
  goal(0) = std::fmod(pos1, 2. * M_PI);
  goal(1) = std::fmod(pos2 + M_PI, 2. * M_PI) - M_PI;
  goal(2) = vel1;
  goal(3) = vel2;
  goal_energy = plant.calculate_total_energy(goal);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < n_x; j++) {
      goal_traj_x[i](j) = goal(j);
    }
    // goal_traj_energy[i] = plant.calculate_total_energy(goal_traj_x[i]);
    goal_traj_energy[i] =
        goal_energy; // save computing time (do not use energy for now)
  }
  for (int i = 0; i < N - 1; i++) {
    for (int j = 0; j < n_u; j++) {
      goal_traj_u[i](j) = 0.; // set default u desired to 0
    }
  }
}

// void ilqr::load_goal_traj(std::string filename){
//     CSVReader reader(filename, ",");
//     std::vector<std::vector<double> > trajectory = reader.getDataDouble(1);
//
//     for (int i=0; i<N; i++){
//         for (int j=0; j<n_x; j++){
//             goal_traj[i](j) = trajectory[i][j];
//         }
//     }
// }

void ilqr::set_goal_traj(double p1[], double p2[], double v1[], double v2[],
                         int from, int to) {
  int c = 0;
  for (int i = from; i < to; i++) {
    goal_traj_x[c](0) = p1[i];
    goal_traj_x[c](1) = p2[i];
    goal_traj_x[c](2) = v1[i];
    goal_traj_x[c](3) = v2[i];
    c += 1;
  }
  for (int i = c; i < N; i++) {
    goal_traj_x[i](0) = p1[to];
    goal_traj_x[i](1) = p2[to];
    goal_traj_x[i](2) = v1[to];
    goal_traj_x[i](3) = v2[to];
  }
}

void ilqr::set_goal_traj(Eigen::Vector<double, n_x> x_tr[], int from, int to) {
  int c = 0;
  for (int i = from; i < to; i++) {
    goal_traj_x[c] = x_tr[i];
    c += 1;
  }
  for (int i = c; i < N; i++) {
    goal_traj_x[i] = x_tr[to];
  }
}

void ilqr::set_goal_traj(Eigen::Vector<double, n_x> x_tr[],
                         Eigen::Vector<double, n_u> u_tr[], int from, int to) {
  int c = 0;
  for (int i = from; i < to; i++) {
    for (int j = 0; j < n_x; j++) {
      goal_traj_x[c](j) = x_tr[i](j);
    }
    c += 1;
  }
  // for (int i=c; i<N; i++){
  //     for (int j=0; j<n_x; j++){
  //         goal_traj_x[i](j) = x_tr[to](j);
  //     }
  // }

  c = 0;
  for (int i = from; i < (to - 1); i++) {
    for (int j = 0; j < n_u; j++) {
      goal_traj_u[c](j) = u_tr[i](j);
    }
    c += 1;
  }
  // for (int i=c; i<N-1; i++){
  //     for (int j=0; j<n_u; j++){
  //         goal_traj_u[i](j) = 0.;
  //     }
  // }
}

void ilqr::set_u_init_traj(double u1[], double u2[]) {
  for (int i = 0; i < N - 1; i++) {
    u_traj[i](0) = u1[i];
    // u_traj[i](1) = u2[i];
    // u1_traj_doubles[i] = u1[i];
    // u2_traj_doubles[i] = u2[i];
  }
  warm_start_u = true;
}

void ilqr::set_u_init_traj(Eigen::Vector<double, n_u> utrj[]) {
  // u_traj = utrj;
  for (int i = 0; i < N - 1; i++) {
    u_traj[i] = utrj[i];
  }
  warm_start_u = true;
}

void ilqr::set_x_init_traj(double p1[], double p2[], double v1[], double v2[]) {
  for (int i = 0; i < N; i++) {
    x_traj[i](0) = p1[i];
    x_traj[i](1) = p2[i];
    x_traj[i](2) = v1[i];
    x_traj[i](3) = v2[i];
    // p1_traj_doubles[i] = p1[i];
    // p2_traj_doubles[i] = p2[i];
    // v1_traj_doubles[i] = v1[i];
    // v2_traj_doubles[i] = v2[i];
  }
  warm_start_x = true;
}

void ilqr::set_x_init_traj(Eigen::Vector<double, n_x> xtrj[]) {
  // x_traj = xtrj;
  for (int i = 0; i < N; i++) {
    x_traj[i] = xtrj[i];
  }
  warm_start_x = true;
}

Eigen::Vector<double, n_x>
ilqr::discrete_dynamics(Eigen::Vector<double, n_x> x,
                        Eigen::Vector<double, n_u> u) {

  // the dp plant has 2 inputs and regulates whether the motors are active with
  // the B matrix the ilqr class should use n_u = 1 to keep the search space as
  // small as possible hence, the broadcasting of u to u_full
  // Eigen::Vector<double, plant.n_u> u_full;
  Eigen::Vector<double, DPPlant::n_u> u_full;
  u_full(0) = 0.;
  u_full(1) = 0.;
  u_full(active_act) = u(0);

  sim.set_state(0.0, x);
  sim.step(u_full, dt, integrator);
  return sim.get_state();
}

double ilqr::stage_cost(Eigen::Vector<double, n_x> x,
                        Eigen::Vector<double, n_u> u, int idx) {
  double eps = 1e-6;
  double pos1_error, pos2_error;
  double vel1_error, vel2_error;
  double u1_cost; //, u2_cost;
  double en_error, scost;
  pos1_error =
      pow((std::fmod(x(0), 2. * M_PI) - goal_traj_x[idx](0) + eps), 2.);
  pos2_error = pow(
      (std::fmod(x(1) + M_PI, 2. * M_PI) - M_PI - goal_traj_x[idx](1) + eps),
      2.);
  vel1_error = pow((x(2) - goal_traj_x[idx](2)), 2.);
  vel2_error = pow((x(3) - goal_traj_x[idx](3)), 2.);
  u1_cost = pow(u(0) - goal_traj_u[idx](0), 2.);
  // std::cout << idx << ", " << u(0) << ", " << goal_traj_u[idx](0) << ", " <<
  // u1_cost << std::endl; u2_cost = pow(u(1), 2.);

  // TODO: acro/pendubot switch
  // double sCu_act;
  // if (active_act == 0){
  //     sCu_act = sCu1;
  // }
  // else{
  //     sCu_act = sCu2;
  // }

  en_error = pow(plant.calculate_total_energy(x) - goal_energy, 2.);
  scost = (sCp1 * pos1_error + sCp2 * pos2_error + sCv1 * vel1_error +
           sCv2 * vel2_error + sCu1 * u1_cost + // sCu2*u2_cost +
           // sCu_act*u1_cost +
           sCen * en_error) /
          (1. * (N - 1));
  return scost;
}

double ilqr::final_cost(Eigen::Vector<double, n_x> x) {
  double eps = 1e-6;
  double pos1_error, pos2_error;
  double vel1_error, vel2_error;
  double en_error, fcost;

  pos1_error = pow((std::fmod(x(0), 2. * M_PI) - goal(0) + eps), 2.);
  pos2_error =
      pow((std::fmod(x(1) + M_PI, 2. * M_PI) - M_PI - goal(1) + eps), 2.);
  vel1_error = pow((x(2) - goal(2)), 2.);
  vel2_error = pow((x(3) - goal(3)), 2.);
  en_error = pow(plant.calculate_total_energy(x) - goal_energy, 2.);
  fcost = fCp1 * pos1_error + fCp2 * pos2_error + fCv1 * vel1_error +
          fCv2 * vel2_error + fCen * en_error;
  return fcost;
}

double ilqr::calculate_cost(bool new_traj) {
  if (verbose > 3) {
    printf("calculate cost\n");
  }
  double total = 0.;
  if (new_traj) {
    for (int i = 0; i < N - 1; i++) {
      total += stage_cost(x_traj_new[i], u_traj_new[i], i); // / (1.*(N-1));
    }
    total += final_cost(x_traj_new[N - 1]);
  } else {
    for (int i = 0; i < N - 1; i++) {
      total += stage_cost(x_traj[i], u_traj[i], i); // / (1.*(N-1));
      // printf("stage cost %e\n", total);
    }
    total += final_cost(x_traj[N - 1]);
  }
  if (verbose > 3) {
    printf("calculate cost done \n");
  }
  return total;
}

void ilqr::compute_dynamics_x(Eigen::Vector<double, n_x> x,
                              Eigen::Vector<double, n_u> u) {
  if (verbose > 3) {
    printf("    compute dynamics x\n");
  }

  // TODO: acro/pendubot switch
  Eigen::Vector<double, DPPlant::n_u> u_full;
  u_full(0) = 0.;
  u_full(1) = 0.;
  u_full(active_act) = u(0);

  Eigen::Matrix<double, n_x, n_x> xd_x;
  Eigen::Matrix<double, n_x, n_x> identity;

  // runge
  Eigen::Vector<double, n_x> k1, k2, k3, k4;

  if (integrator == "euler") {
    xd_x = plant.get_dynx(x, u_full);
  } else {
    k1 = plant.rhs(0., x, u_full);
    k2 = plant.rhs(0., x + 0.5 * dt * k1, u_full);
    k3 = plant.rhs(0., x + 0.5 * dt * k2, u_full);
    // k4 = plant.rhs(0., x+dt*k3, u);

    xd_x = (plant.get_dynx(x, u_full) +
            2 * plant.get_dynx(x + 0.5 * dt * k1, u_full) +
            2 * plant.get_dynx(x + 0.5 * dt * k2, u_full) +
            plant.get_dynx(x + dt * k3, u_full)) /
           6.0;
  }
  identity = Eigen::MatrixXd::Identity(n_x, n_x);
  dyn_x = identity + xd_x * dt;
}

void ilqr::compute_dynamics_u(Eigen::Vector<double, n_x> x,
                              Eigen::Vector<double, n_u> u) {
  if (verbose > 3) {
    printf("    compute dynamics u\n");
  }

  Eigen::Vector<double, DPPlant::n_u> u_full;
  u_full(0) = 0.;
  u_full(1) = 0.;
  u_full(active_act) = u(0);

  Eigen::Matrix<double, n_x, DPPlant::n_u> dynu = plant.get_dynu(x, u_full);

  for (int i = 0; i < n_x; i++) {
    dyn_u(i, 0) = dynu(i, active_act) * dt; // acrobot
  }
}

void ilqr::compute_stage_x(Eigen::Vector<double, n_x> x,
                           Eigen::Vector<double, n_u> u, int idx) {
  if (verbose > 3) {
    printf("    compute stage x\n");
  }

  double eps = 1e-6;
  en_diff = plant.calculate_total_energy(x) - goal_traj_energy[idx];
  Eigen::Vector<double, n_x> E_x = plant.get_Ex(x);

  stage_x(0) =
      (2. * sCp1 * (std::fmod(x(0), 2. * M_PI) - goal_traj_x[idx](0) + eps) +
       2. * sCen * en_diff * E_x(0)) /
      (1. * (N - 1));
  stage_x(1) = (2. * sCp2 *
                    (std::fmod(x(1) + M_PI, 2. * M_PI) - M_PI -
                     goal_traj_x[idx](1) + eps) +
                2. * sCen * en_diff * E_x(1)) /
               (1. * (N - 1));
  stage_x(2) = (2. * sCv1 * (x(2) - goal_traj_x[idx](2)) +
                2. * sCen * en_diff * E_x(2)) /
               (1. * (N - 1));
  stage_x(3) = (2. * sCv2 * (x(3) - goal_traj_x[idx](3)) +
                2. * sCen * en_diff * E_x(3)) /
               (1. * (N - 1));
}

void ilqr::compute_stage_u(Eigen::Vector<double, n_x> x,
                           Eigen::Vector<double, n_u> u, int idx) {
  if (verbose > 3) {
    printf("    compute stage u\n");
  }
  // double sCu_act;
  // if (active_act == 0){
  //     sCu_act = sCu1;
  // }
  // else{
  //     sCu_act = sCu2;
  // }
  stage_u(0) = 2 * sCu1 * (u(0) - goal_traj_u[idx](0)) / (1. * (N - 1));
  // stage_u(1) = 2.*sCu2*u(1);
  // stage_u(0) = 2*sCu_act*u(0);
}

void ilqr::compute_stage_xx(Eigen::Vector<double, n_x> x,
                            Eigen::Vector<double, n_u> u, int idx) {
  if (verbose > 3) {
    printf("    compute stage xx\n");
  }

  en_diff = plant.calculate_total_energy(x) - goal_traj_energy[idx];
  Eigen::Vector<double, n_x> E_x = plant.get_Ex(x);
  Eigen::Matrix<double, n_x, n_x> E_xx = plant.get_Exx(x);

  for (int i = 0; i < n_x; i++) {
    for (int j = 0; j < n_x; j++) {
      stage_xx(i, j) = 2 * sCen * (E_xx(i, j) * en_diff + E_x(i) * E_x(j));
    }
  }

  stage_xx(0, 0) += 2. * sCp1 / (1. * (N - 1));
  stage_xx(1, 1) += 2. * sCp2 / (1. * (N - 1));
  stage_xx(2, 2) += 2. * sCv1 / (1. * (N - 1));
  stage_xx(3, 3) += 2. * sCv2 / (1. * (N - 1));
}

void ilqr::compute_stage_ux(Eigen::Vector<double, n_x> x,
                            Eigen::Vector<double, n_u> u, int idx) {
  if (verbose > 3) {
    printf("    compute stage ux\n");
  }
  stage_ux(0, 0) = 0.;
  stage_ux(0, 1) = 0.;
  stage_ux(0, 2) = 0.;
  stage_ux(0, 3) = 0.;
}

void ilqr::compute_stage_uu(Eigen::Vector<double, n_x> x,
                            Eigen::Vector<double, n_u> u, int idx) {
  if (verbose > 3) {
    printf("    compute stage uu\n");
  }
  // TODO: acro-/pendubot switch
  // double sCu_act;
  // if (active_act == 0){
  //     sCu_act = sCu1;
  // }
  // else{
  //     sCu_act = sCu2;
  // }
  stage_uu(0, 0) = 2. * sCu1 / (1. * (N - 1));
  // stage_uu(0,1) = 0.;
  // stage_uu(1,0) = 0.;
  // stage_uu(1,1) = 2.*sCu2;
  // stage_uu(0,0) = 2.*sCu_act;
}

void ilqr::compute_final_x(Eigen::Vector<double, n_x> x) {
  if (verbose > 3) {
    printf("    compute final x\n");
  }
  double eps = 1e-6;
  en_diff = plant.calculate_total_energy(x) - goal_energy;
  Eigen::Vector<double, n_x> E_x = plant.get_Ex(x);

  final_x(0) = 2. * fCp1 * (std::fmod(x(0), 2. * M_PI) - goal(0) + eps) +
               2. * fCen * en_diff * E_x(0);
  final_x(1) =
      2. * fCp2 * (std::fmod(x(1) + M_PI, 2. * M_PI) - M_PI - goal(1) + eps) +
      2. * fCen * en_diff * E_x(1);
  final_x(2) = 2. * fCv1 * (x(2) - goal(2)) + 2. * fCen * en_diff * E_x(2);
  final_x(3) = 2. * fCv2 * (x(3) - goal(3)) + 2. * fCen * en_diff * E_x(3);

  if (verbose > 3) {
    std::cout << "final_x " << final_x << std::endl;
  }
}

void ilqr::compute_final_xx(Eigen::Vector<double, n_x> x) {
  if (verbose > 3) {
    printf("    compute final xx\n");
  }
  en_diff = plant.calculate_total_energy(x) - goal_energy;
  Eigen::Vector<double, n_x> E_x = plant.get_Ex(x);
  Eigen::Matrix<double, n_x, n_x> E_xx = plant.get_Exx(x);

  for (int i = 0; i < n_x; i++) {
    for (int j = 0; j < n_x; j++) {
      final_xx(i, j) = 2. * fCen * (E_xx(i, j) * en_diff + E_x(i) * E_x(j));
    }
  }

  final_xx(0, 0) += 2. * fCp1;
  final_xx(1, 1) += 2. * fCp2;
  final_xx(2, 2) += 2. * fCv1;
  final_xx(3, 3) += 2. * fCv2;
  if (verbose > 3) {
    std::cout << "final_xx " << final_xx << std::endl;
  }
}

void ilqr::compute_derivatives(Eigen::Vector<double, n_x> x,
                               Eigen::Vector<double, n_u> u, int idx) {
  if (verbose > 3) {
    printf("compute derivatives\n");
  }

  // en_diff = plant.calculate_total_energy(x) - goal_energy;
  compute_dynamics_x(x, u);
  compute_dynamics_u(x, u);
  compute_stage_x(x, u, idx);
  compute_stage_u(x, u, idx);
  compute_stage_xx(x, u, idx);
  compute_stage_ux(x, u, idx);
  compute_stage_uu(x, u, idx);
  // compute_final_x(x);
  // compute_final_xx(x);

  if (verbose > 4) {
    std::cout << "derivative terms results" << std::endl
              << "dyn_x " << dyn_x << std::endl
              << "dyn_u " << dyn_u << std::endl
              << "stage_x " << stage_x << std::endl
              << "stage_u " << stage_u << std::endl
              << "stage_xx " << stage_xx << std::endl
              << "stage_ux " << stage_ux << std::endl
              << "stage_uu " << stage_uu
              << std::endl
              //<< "final_x " << final_x << std::endl
              //<< "final_xx " << final_xx << std::endl;
              << std::endl;
  }
}

void ilqr::rollout() {
  if (verbose > 2) {
    printf("rollout\n");
  }
  x_traj[0] = x0;
  for (int i = 0; i < N - 1; i++) {
    x_traj[i + 1] = discrete_dynamics(x_traj[i], u_traj[i]);
  }
}

void ilqr::calculate_Q_terms() {
  if (verbose > 3) {
    printf("calculate Q terms\n");
    printf("inputs: ");
    std::cout << "      stage_x" << stage_x << std::endl;
    std::cout << "      dyn_x" << dyn_x << std::endl;
    std::cout << "      V_x" << V_x << std::endl;
  }

  Q_x = stage_x + dyn_x.transpose() * V_x;
  Q_u = stage_u + dyn_u.transpose() * V_x;
  Q_xx = stage_xx + dyn_x.transpose() * (V_xx * dyn_x);
  Q_ux = stage_ux + dyn_u.transpose() * (V_xx * dyn_x);
  Q_uu = stage_uu + dyn_u.transpose() * (V_xx * dyn_u);
  if (verbose > 4) {
    std::cout << "calculate Q terms results" << std::endl
              << "Q_x " << Q_x << std::endl
              << "Q_u " << Q_u << std::endl
              << "Q_xx " << Q_xx << std::endl
              << "Q_ux " << Q_ux << std::endl
              << "Q_uu " << Q_uu << std::endl
              << std::endl;
  }
}

void ilqr::calculate_gains(double regu) {
  if (verbose > 3) {
    printf("    calculate gains\n");
  }

  Eigen::Matrix<double, n_u, n_u> Q_uu_regu_inv;
  Q_uu_regu_inv(0, 0) = 1. / (Q_uu(0, 0) + regu);

  k = -Q_uu_regu_inv * Q_u;
  K = -Q_uu_regu_inv * Q_ux;
  if (verbose > 4) {
    std::cout << "Gains results" << std::endl
              << "k " << k << std::endl
              << "K " << K << std::endl
              << std::endl;
  }
}

void ilqr::calculate_V_terms() {
  if (verbose > 3) {
    printf("    calculate V terms\n");
  }
  // V_x = Q_x - K.transpose()*Q_uu*k;
  // V_xx = Q_xx - K.transpose()*Q_uu*K;
  V_x = Q_x + K.transpose() * Q_u +
        0.5 * (Q_ux.transpose() * k + K.transpose() * (Q_uu.transpose() * k));
  V_xx = Q_xx + Q_ux.transpose() * K + K.transpose() * Q_ux +
         K.transpose() * Q_uu * K;
  if (verbose > 4) {
    std::cout << "calculate V terms results" << std::endl
              << "V_x " << V_x << std::endl
              << "V_xx " << V_xx << std::endl
              << std::endl;
  }
}

double ilqr::expected_cost_reduction() {
  if (verbose > 3) {
    printf("    expected cost reduction\n");
  }
  // printf("expected_cost_redu: Q_u %f, Q_uu %f, k %f\n", Q_u, Q_uu, k);
  return (-Q_u.transpose() * k - 0.5 * k.transpose() * Q_uu * k)(0);
}

void ilqr::forward_pass() {
  if (verbose > 3) {
    printf("forward pass\n");
  }
  // Eigen::Vector2d xdiff;
  x_traj_new[0] = x_traj[0];
  for (int i = 0; i < N - 1; i++) {
    // xdiff[0] = x_traj_new[i].pos - x_traj[i].pos;
    // xdiff[1] = x_traj_new[i].vel - x_traj[i].vel;
    // u_traj_new[i].u1 = u_traj[i].u1 + k_traj[i] +
    // K_traj[i].transpose()*xdiff; x_traj_new[i+1] =
    // pendulum_discrete_dynamics(x_traj_new[i], u_traj_new[i]);

    u_traj_new[i] =
        u_traj[i] + k_traj[i] + K_traj[i] * (x_traj_new[i] - x_traj[i]);
    x_traj_new[i + 1] = discrete_dynamics(x_traj_new[i], u_traj_new[i]);
  }
}

double ilqr::backward_pass(double regu) {
  if (verbose > 3) {
    printf("backward pass\n");
  }
  double expected_cost_redu = 0.0;
  compute_final_x(x_traj[N - 1]);
  compute_final_xx(x_traj[N - 1]);
  V_x = final_x;
  V_xx = final_xx;
  for (int i = N - 2; i > -1; i--) { // TODO: i>0 or i>-1? i=N-1 or i=N-2?
    // std::cout << "x " << x_traj[i] << ". u " << u_traj[i] << std::endl;
    compute_derivatives(x_traj[i], u_traj[i], i);
    calculate_Q_terms();
    calculate_gains(regu);
    k_traj[i] = k;
    K_traj[i] = K;
    calculate_V_terms();
    expected_cost_redu += expected_cost_reduction();
    if (verbose > 4) {
      printf("bw pass step %d expected_cost_redu %f\n", i, expected_cost_redu);
    }
  }
  return expected_cost_redu;
}

void ilqr::run_ilqr(int max_iter, double break_cost_redu, double regu_init,
                    double max_regu, double min_regu) {
  if (not warm_start_u) {
    for (int i = 0; i < N - 1; i++) {
      for (int j = 0; j < n_u; j++) {
        // u_traj[i](j) = 0.0; //((double) rand() / RAND_MAX)*0.0001;
        u_traj[i](j) = ((double)rand() / RAND_MAX) * 0.0001;
      }
    }
  }
  if (not warm_start_x) {
    rollout();
  }

  if (verbose > 1) {
    std::cout << "############################\n "
                 "Parameters\n############################\n"
              << std::endl;
    std::cout << "mass1 " << mass1 << std::endl;
    std::cout << "mass2 " << mass2 << std::endl;
    std::cout << "length1 " << length1 << std::endl;
    std::cout << "length2 " << length2 << std::endl;
    std::cout << "damping1 " << damping1 << std::endl;
    std::cout << "damping2 " << damping2 << std::endl;
    std::cout << "coublomb_friction1 " << coulomb_friction1 << std::endl;
    std::cout << "coublomb_friction2 " << coulomb_friction2 << std::endl;
    std::cout << "gravity " << gravity << std::endl;
    std::cout << "inertia1 " << inertia1 << std::endl;
    std::cout << "inertia2 " << inertia2 << std::endl;
    std::cout << "torque_limit1 " << torque_limit1 << std::endl;
    std::cout << "torque_limit2 " << torque_limit2 << std::endl;
    std::cout << "dt " << dt << std::endl;
    std::cout << "integrator " << integrator << std::endl;
    std::cout << "start_pos1 " << x0(0) << std::endl;
    std::cout << "start_pos2 " << x0(1) << std::endl;
    std::cout << "start_vel1 " << x0(2) << std::endl;
    std::cout << "start_vel2 " << x0(3) << std::endl;
    std::cout << "goal_pos1 " << goal(0) << std::endl;
    std::cout << "goal_pos2 " << goal(1) << std::endl;
    std::cout << "goal_vel1 " << goal(2) << std::endl;
    std::cout << "goal_vel2 " << goal(3) << std::endl;
    std::cout << "sCu1 " << sCu1 << std::endl;
    std::cout << "sCu2 " << sCu2 << std::endl;
    std::cout << "sCp1 " << sCp1 << std::endl;
    std::cout << "sCp2 " << sCp2 << std::endl;
    std::cout << "sCv1 " << sCv1 << std::endl;
    std::cout << "sCv2 " << sCv2 << std::endl;
    std::cout << "sCen " << sCen << std::endl;
    std::cout << "fCp1 " << fCp1 << std::endl;
    std::cout << "fCp2 " << fCp2 << std::endl;
    std::cout << "fCv1 " << fCv1 << std::endl;
    std::cout << "fCv2 " << fCv2 << std::endl;
    std::cout << "fCen " << fCen << std::endl;
    std::cout << "max_iter " << max_iter << std::endl;
    std::cout << "break_cost_redu " << break_cost_redu << std::endl;
    std::cout << "regu_init " << regu_init << std::endl;
    std::cout << "max_regu " << max_regu << std::endl;
    std::cout << "min_regu " << min_regu << std::endl;
    std::cout << "N " << N << std::endl;
    std::cout << "\n\n";
  }

  double last_cost = calculate_cost(false);
  double total_cost;

  double regu = regu_init;
  double expected_cost_redu = 0.;

  for (int n = 0; n < max_iter; n++) {
    expected_cost_redu = backward_pass(regu);
    forward_pass();
    total_cost = calculate_cost(true);

    if (verbose > 2) {
      printf("iteration %d, ", n);
      printf("last_pos1 %f, ", x_traj_new[N - 1](0));
      printf("last_pos2 %f, ", x_traj_new[N - 1](1));
      printf("last_cost %e, ", last_cost);
      printf("total_cost %e\n", total_cost);
    }
    if ((last_cost - total_cost) > 0.) {
      // improvement
      if (verbose > 1) {
        printf("improvement");
      }
      // printf("y ");
      last_cost = total_cost;
      for (int i = 0; i < N; i++) {
        x_traj[i] = x_traj_new[i];
        best_k_traj[i] = k_traj[i];
        best_K_traj[i] = K_traj[i];
      }
      for (int i = 0; i < N - 1; i++) {
        u_traj[i] = u_traj_new[i];
      }
      regu *= 0.7;
      // regu *= 0.9;
    } else {
      // no improvement
      if (verbose > 1) {
        printf("noimprovement");
      }
      // printf("n ");
      if (regu >= max_regu) {
        if (verbose > 1) {
          printf("\nReached max regularization -> Stopping\n");
        }
        break;
      }
      regu *= 2.0;
      // regu *= 1.1;
    }
    if (regu < min_regu) {
      regu = min_regu;
    }
    if (regu > max_regu) {
      regu = max_regu;
    }
    if (verbose > 2) {
      printf(" regu %f", regu);
      // printf("expected cost redu %f, regu: %f\n", expected_cost_redu, regu);
    }
    if (expected_cost_redu < break_cost_redu) {
      if (verbose > 1) {
        printf("No further cost reduction expected -> Stopping\n");
      }
      break;
    }
    if (verbose > 2) {
      printf("\n");
    }
  }
}

double *ilqr::get_u1_traj() {
  if (active_act == 0) {
    for (int i = 0; i < N - 1; i++) {
      u1_traj_doubles[i] = u_traj[i](0);
    }
  } else {
    for (int i = 0; i < N - 1; i++) {
      u1_traj_doubles[i] = 0.;
    }
  }
  return u1_traj_doubles;
}

double *ilqr::get_u2_traj() {
  if (active_act == 1) {
    for (int i = 0; i < N - 1; i++) {
      u2_traj_doubles[i] = u_traj[i](0);
    }
  } else {
    for (int i = 0; i < N - 1; i++) {
      u2_traj_doubles[i] = 0.;
    }
  }
  return u2_traj_doubles;
}

double *ilqr::get_p1_traj() {
  for (int i = 0; i < N; i++) {
    p1_traj_doubles[i] = x_traj[i](0);
  }
  return p1_traj_doubles;
}

double *ilqr::get_p2_traj() {
  for (int i = 0; i < N; i++) {
    p2_traj_doubles[i] = x_traj[i](1);
  }
  return p2_traj_doubles;
}

double *ilqr::get_v1_traj() {
  for (int i = 0; i < N; i++) {
    v1_traj_doubles[i] = x_traj[i](2);
  }
  return v1_traj_doubles;
}

double *ilqr::get_v2_traj() {
  for (int i = 0; i < N; i++) {
    v2_traj_doubles[i] = x_traj[i](3);
  }
  return v2_traj_doubles;
}

Eigen::Vector<double, n_u> *ilqr::get_u_traj() { return u_traj; }

Eigen::Vector<double, n_x> *ilqr::get_x_traj() { return x_traj; }

void ilqr::save_trajectory_csv() {
  std::string filename = "trajectory.csv";
  save_trajectory_csv(filename);
}

void ilqr::save_trajectory_csv(std::string filename) {

  std::ofstream traj_file;
  traj_file.open(filename);
  traj_file << "time,pos1,pos2,vel1,vel2,tau1,tau2,K11,K12,K13,K14,K21,K22,K23,"
               "K24,k1,k2\n";

  for (int i = 0; i < N - 1; i++) {
    traj_file << dt * i << ", ";
    for (int j = 0; j < n_x; j++) {
      traj_file << x_traj[i](j) << ", ";
    }
    if (active_act == 0) {
      traj_file << u_traj[i](0) << ", ";
      traj_file << 0.0 << ", ";
      traj_file << best_K_traj[i](0, 0) << ", ";
      traj_file << best_K_traj[i](0, 1) << ", ";
      traj_file << best_K_traj[i](0, 2) << ", ";
      traj_file << best_K_traj[i](0, 3) << ", ";
      traj_file << 0.0 << ", ";
      traj_file << 0.0 << ", ";
      traj_file << 0.0 << ", ";
      traj_file << 0.0 << ", ";
      traj_file << best_k_traj[i](0) << ", ";
      traj_file << 0.0 << "\n";
    } else if (active_act == 1) {
      traj_file << 0.0 << ", ";
      traj_file << u_traj[i](0) << ", ";
      traj_file << 0.0 << ", ";
      traj_file << 0.0 << ", ";
      traj_file << 0.0 << ", ";
      traj_file << 0.0 << ", ";
      traj_file << best_K_traj[i](0, 0) << ", ";
      traj_file << best_K_traj[i](0, 1) << ", ";
      traj_file << best_K_traj[i](0, 2) << ", ";
      traj_file << best_K_traj[i](0, 3) << ", ";
      traj_file << 0.0 << ", ";
      traj_file << best_k_traj[i](0) << "\n";
    }
  }

  // for(int i=0; i<N-1; i++){
  //     traj_file << dt*i << ", "
  //               << x_traj[i](0) << ", "
  //               << x_traj[i](1) << ", "
  //               << x_traj[i](2) << ", "
  //               << x_traj[i](3) << ", "
  //               << 0.0 << ", "
  //               << u_traj[i](0) << ", "
  //               << 0.0 << ", "
  //               << 0.0 << ", "
  //               << 0.0 << ", "
  //               << 0.0 << ", "
  //	  << best_K_traj[i](0,0) << ", "
  //	  << best_K_traj[i](0,1) << ", "
  //	  << best_K_traj[i](0,2) << ", "
  //	  << best_K_traj[i](0,3) << ", "
  //	  << 0.0 << ", "
  //	  << best_k_traj[i](0) << "\n";
  //}
  // traj_file << (N-1)*dt << ", "
  //          << x_traj[N-1](0) << ", "
  //          << x_traj[N-1](1) << ", "
  //          << x_traj[N-1](2) << ", "
  //          << x_traj[N-1](3) << ", "
  //          << 0.0 << ", "
  //          << 0.0 << ", "
  //          << 0.0 << ", "
  //          << 0.0 << ", "
  //          << 0.0 << ", "
  //          << 0.0 << ", "
  //          << K_traj[N-1](0,0) << ", "
  //          << K_traj[N-1](0,1) << ", "
  //          << K_traj[N-1](0,2) << ", "
  //          << K_traj[N-1](0,3) << ", "
  //          << 0.0 << ", "
  //          << k_traj[N-1](0);
  traj_file.close();
}
