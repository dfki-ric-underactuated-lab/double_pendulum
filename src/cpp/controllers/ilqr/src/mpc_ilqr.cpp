#include <iostream>
#include <fstream>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "../../../utils/src/csv_reader.hpp"
#include "../../../model/src/dp_plant.hpp"
#include "../../../simulator/src/simulator.hpp"
#include "ilqr.hpp"
#include "ilqr_mpc.hpp"


int main(int argc, char *argv[], char *envp[]){

    std::string configfile = "config.yml";
    std::string trajfile = "trajectory.csv";
    std::string foldername = "data";
    if (argc >= 1){configfile = std::string(argv[1]);}
    if (argc >= 2){trajfile = std::string(argv[2]);}
    if (argc >= 3){foldername = std::string(argv[3]);}

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
    double T = 10.0;
    int integrator_ind = 1;
    std::string integrator = "runge_kutta";

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
    int max_iter = 2;
    double break_cost_redu = 1e-6;
    double regu_init = 100.;
    int N=1000;
    int N_init=1000;

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
    if (config["T"]) {T=config["T"].as<double>();}
    if (config["integrator"]){integrator_ind = config["integrator"].as<int>();}
    if (config["integrator"]){
        int integrator_ind = config["integrator"].as<int>();
        if(integrator_ind == 0){
            integrator = "euler";
        }
        else{
            integrator = "runge_kutta";
        }
    }
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
    if (config["Ninit"]) {N_init=config["Ninit"].as<int>();}

    if (config["verbose"]) {verbose=config["verbose"].as<int>();}

    if (config["max_iter"]) {max_iter=config["max_iter"].as<int>();}
    if (config["break_cost_redu"]) {break_cost_redu=config["break_cost_redu"].as<double>();}
    if (config["regu_init"]) {regu_init=config["regu_init"].as<double>();}

    // plant and simulator
    DPPlant plant = DPPlant(false, true);
    plant.set_parameters(mass1, mass2,
                         length1, length2,
                         com1, com2,
                         inertia1, inertia2,
                         damping1, damping2,
                         coulomb_friction1, coulomb_friction2,
                         gravity,
                         torque_limit1, torque_limit2);
    Simulator sim = Simulator();
    sim.set_plant(plant);

    //set start conditions
    Eigen::Vector<double, ilqr::n_x> state;
    state(0) = start_pos1;
    state(1) = start_pos2;
    state(2) = start_vel1;
    state(3) = start_vel2;
    //sim.set_state(0.0, state);
    Eigen::Vector<double, ilqr::n_x> goal;
    goal(0) = goal_pos1;
    goal(1) = goal_pos2;
    goal(2) = goal_vel1;
    goal(3) = goal_vel2;

    // load initial trajectory
    CSVReader reader(trajfile, ",");
    std::vector<std::vector<double> > trajectory = reader.getDataDouble(1);
    int TN = trajectory.size();

    Eigen::Vector<double, ilqr::n_u>* u_traj = new Eigen::Vector<double, ilqr::n_u>[TN-1];
    Eigen::Vector<double, ilqr::n_x>* x_traj = new Eigen::Vector<double, ilqr::n_x>[TN];
    //int Nmin = std::min(N_init, TN);

    for (int i=0; i<TN-1; i++){
        u_traj[i](0) = trajectory[i][5];
    }
    for (int i=0; i<TN; i++){
        x_traj[i](0) = trajectory[i][0];
        x_traj[i](1) = trajectory[i][1];
        x_traj[i](2) = trajectory[i][2];
        x_traj[i](3) = trajectory[i][3];
    }

    int n_steps = (int) (T / dt);
    //Eigen::Vector<double, ilqr::n_u> u;
    Eigen::Vector<double, DPPlant::n_u> u_full;
    //u_full(0) = 0.;
    //u_full(1) = 0.;

    //ilqr ilqr_calc(N);
    //ilqr_calc.read_parameter_file(configfile);
    //ilqr_mpc ilmpc = ilqr_mpc();
    ilqr_mpc ilmpc = ilqr_mpc(N, TN);
    ilmpc.read_parameter_file(configfile);
    //ilmpc.set_start(state);
    ilmpc.set_goal(goal);
    ilmpc.set_u_init_traj(u_traj);
    ilmpc.set_x_init_traj(x_traj);


    std::ofstream traj_file;
    traj_file.open (foldername+"/trajectory_mpc.csv");
    traj_file << "time, pos1, pos2, vel1, vel2, tau1, tau2\n";

    for (int s=0; s<n_steps; s++){
        u_full(0) = 0.;
        u_full(1) = ilmpc.get_control_output(state); //(0);
        sim.set_state(0.0, state);
        sim.step(u_full, dt, integrator);
        state = sim.get_state();

        std::cout << "Step " << s << " ";
        std::cout << state(0) << ", ";
        std::cout << state(1) << ", ";
        std::cout << state(2) << ", ";
        std::cout << state(3) << ", ";
        std::cout << u_full(0) << ", ";
        std::cout << u_full(1);
        if (s<TN){
            std::cout << ", (" << trajectory[s][5] << ")";
            if(pow(pow(u_full(1) - trajectory[s][5], 2.0), 0.5) > 0.01){
                std::cout << " <-------------";
            }
        }
        std::cout << std::endl;

        traj_file << dt*s << ", "
                  << state(0) << ", "
                  << state(1) << ", "
                  << state(2) << ", "
                  << state(3) << ", "
                  << u_full(0) << ", "
                  << u_full(1) << "\n";
    }
    traj_file.close();
}

