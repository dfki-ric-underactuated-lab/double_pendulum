#include <iostream>
#include <fstream>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "ilqr.hpp"


int main(int argc, char *argv[], char *envp[]){

    std::string configfile = "config.yml";
    std::string foldername = "data";
    if (argc >= 1){configfile = std::string(argv[1]);}
    if (argc >= 2){foldername = std::string(argv[2]);}

    // default parameters

    int verbose = 0;

    //pendulum parameters
    double mass1 = 0.5;
    double mass2 = 0.5;
    double length1 = 0.5;
    double length2 = 0.5;
    double com1 = 0.5;
    double com2 = 0.5;
    double inertia1 = mass1*com1*com1;
    double inertia2 = mass2*com2*com2;
    double damping1 = 0.0;
    double damping2 = 0.0;
    double coulomb_friction1 = 0.0;
    double coulomb_friction2 = 0.0;
    double gravity = 9.81;
    double torque_limit1 = 0.0;
    double torque_limit2 = 10.0;

    // simulation parameters
    double dt = 0.01;
    int integrator_ind = 1;

    //swingup parameters
    double start_pos1 = 0.0;
    double start_pos2 = 0.0;
    double start_vel1 = 0.0;
    double start_vel2 = 0.0;
    double goal_pos1 = 3.1415;
    double goal_pos2 = 0.0;
    double goal_vel1 = 0.0;
    double goal_vel2 = 0.0;

    // cost parameters
    double sCu1 = 0.005;
    double sCu2 = 0.005;
    double sCp1 = 0.0;
    double sCp2 = 0.0;
    double sCv1 = 0.0;
    double sCv2 = 0.0;
    double sCen = 0.0;
    double fCp1 = 1000.;
    double fCp2 = 1000.;
    double fCv1 = 10.0;
    double fCv2 = 10.0;
    double fCen = 0.0;

    // other ilqr parameters
    int max_iter = 100;
    double break_cost_redu = 1e-6;
    double regu_init = 100.;
    double max_regu = 10000.;
    double min_regu = 0.01;
    int N=1000;

    //TODO fix yaml loading
    // read parameters from yaml file
    YAML::Node config = YAML::LoadFile(configfile);
    if (config["mass1"]) {mass1=config["mass1"].as<double>();}
    if (config["mass2"]) {mass2=config["mass2"].as<double>();}
    if (config["length1"]) {length1=config["length1"].as<double>();}
    if (config["length2"]) {length2=config["length2"].as<double>();}
    if (config["com1"]) {com1=config["com1"].as<double>();}
    if (config["com2"]) {com2=config["com2"].as<double>();}
    if (config["damping1"]) {damping1=config["damping1"].as<double>();}
    if (config["damping2"]) {damping2=config["damping2"].as<double>();}
    if (config["coulomb_friction1"]) {coulomb_friction1=config["coulomb_friction1"].as<double>();}
    if (config["coulomb_friction2"]) {coulomb_friction2=config["coulomb_friction2"].as<double>();}
    if (config["gravity"]) {gravity=config["gravity"].as<double>();}
    if (config["inertia1"]){
        inertia1=config["inertia1"].as<double>();
    }
    else{
        inertia1 = mass1*com1*com1;
    }
    if (config["inertia2"]){
        inertia2=config["inertia2"].as<double>();
    }
    else{
        inertia2 = mass2*com2*com2;
    }
    if (config["torque_limit1"]) {torque_limit1=config["torque_limit1"].as<double>();}
    if (config["torque_limit2"]) {torque_limit2=config["torque_limit2"].as<double>();}
    if (config["deltaT"]) {dt=config["deltaT"].as<double>();}
    if (config["integrator"]){integrator_ind = config["integrator"].as<int>();}
    if (config["start_pos1"]) {start_pos1=config["start_pos1"].as<double>();}
    if (config["start_pos2"]) {start_pos2=config["start_pos2"].as<double>();}
    if (config["start_vel1"]) {start_vel1=config["start_vel1"].as<double>();}
    if (config["start_vel2"]) {start_vel2=config["start_vel2"].as<double>();}
    if (config["goal_pos1"]) {goal_pos1=config["goal_pos1"].as<double>();}
    if (config["goal_pos2"]) {goal_pos2=config["goal_pos2"].as<double>();}
    if (config["goal_vel1"]) {goal_vel1=config["goal_vel1"].as<double>();}
    if (config["goal_vel2"]) {goal_vel2=config["goal_vel2"].as<double>();}
    if (config["sCu1"]) {sCu1=config["sCu1"].as<double>();}
    if (config["sCu2"]) {sCu2=config["sCu2"].as<double>();}
    if (config["sCp1"]) {sCp1=config["sCp1"].as<double>();}
    if (config["sCp2"]) {sCp2=config["sCp2"].as<double>();}
    if (config["sCv1"]) {sCv1=config["sCv1"].as<double>();}
    if (config["sCv2"]) {sCv2=config["sCv2"].as<double>();}
    if (config["sCen"]) {sCen=config["sCen"].as<double>();}
    if (config["fCp1"]) {fCp1=config["fCp1"].as<double>();}
    if (config["fCp2"]) {fCp2=config["fCp2"].as<double>();}
    if (config["fCv1"]) {fCv1=config["fCv1"].as<double>();}
    if (config["fCv2"]) {fCv2=config["fCv2"].as<double>();}
    if (config["fCen"]) {fCen=config["fCen"].as<double>();}
    if (config["N"]) {N=config["N"].as<int>();}

    if (config["verbose"]) {verbose=config["verbose"].as<int>();}

    if (config["max_iter"]) {max_iter=config["max_iter"].as<int>();}
    if (config["break_cost_redu"]) {break_cost_redu=config["break_cost_redu"].as<double>();}
    if (config["regu_init"]) {regu_init=config["regu_init"].as<double>();}
    if (config["max_regu"]) {max_regu=config["max_regu"].as<double>();}
    if (config["min_regu"]) {min_regu=config["min_regu"].as<double>();}

    ilqr ilqr_calc(N);

    ilqr_calc.set_verbose(verbose);

    ilqr_calc.set_parameters(integrator_ind, dt);
    ilqr_calc.set_model_parameters(mass1, mass2,
                                   length1, length2,
                                   com1, com2,
                                   inertia1, inertia2,
                                   damping1, damping2,
                                   coulomb_friction1, coulomb_friction2,
                                   gravity,
                                   torque_limit1, torque_limit2);
    ilqr_calc.set_cost_parameters(sCu1, sCu2,
                                  sCp1, sCp2,
                                  sCv1, sCv2,
                                  sCen,
                                  fCp1, fCp2,
                                  fCv1, fCv2,
                                  fCen);


    ilqr_calc.set_start(start_pos1, start_pos2,
                        start_vel1, start_vel2);
    ilqr_calc.set_goal(goal_pos1, goal_pos2,
                       goal_vel1, goal_vel2);

    ilqr_calc.run_ilqr(max_iter, break_cost_redu, regu_init, max_regu, min_regu);

    ilqr_calc.save_trajectory_csv();
    //double* u1_traj_doubles = new double[N-1];
    //double* u2_traj_doubles = new double[N-1];
    //double* p1_traj_doubles = new double[N];
    //double* p2_traj_doubles = new double[N];
    //double* v1_traj_doubles = new double[N];
    //double* v2_traj_doubles = new double[N];

    //u1_traj_doubles = ilqr_calc.get_u1_traj();
    //u2_traj_doubles = ilqr_calc.get_u2_traj();
    //p1_traj_doubles = ilqr_calc.get_p1_traj();
    //p2_traj_doubles = ilqr_calc.get_p2_traj();
    //v1_traj_doubles = ilqr_calc.get_v1_traj();
    //v2_traj_doubles = ilqr_calc.get_v2_traj();

    //std::ofstream traj_file;
    //traj_file.open (foldername+"/trajectory.csv");
    //traj_file << "pos, vel, tau\n";

    //for (int i=0; i<N-1; i++){
    //    traj_file << ilqr_calc.x_traj[i](0) << ", " << ilqr_calc.x_traj[i](1) << ", " << ilqr_calc.x_traj[i](2) << ", " << ilqr_calc.x_traj[i](3) << ", " << 0.0 << ", " << ilqr_calc.u_traj[i](0) << "\n";
    //}

    //Eigen::Vector<double, ilqr_calc.n_x> xf = ilqr_calc.x_traj[N-1];
    //printf("Final state %f %f %f %f \n", xf(0), xf(1), xf(2), xf(3));
    //traj_file << ilqr_calc.x_traj[N-1](0) << ", " << ilqr_calc.x_traj[N-1](1) << ", "<< ilqr_calc.x_traj[N-1](2) << ", " << ilqr_calc.x_traj[N-1](3) << ", " << 0.0 << ", " << 0.0;
    // printf("Random state %f %f %f %f \n", p1_traj_doubles[4],
    //                                       p2_traj_doubles[4],
    //                                       v1_traj_doubles[4],
    //                                       v2_traj_doubles[4]);

    // delete [] u1_traj_doubles;
    // delete [] u2_traj_doubles;
    // delete [] p1_traj_doubles;
    // delete [] p2_traj_doubles;
    // delete [] v1_traj_doubles;
    // delete [] v2_traj_doubles;

}

