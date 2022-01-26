#include <iostream>
#include <fstream>
#include <sstream>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "../../utils/src/csv_reader.hpp"
#include "../../model/src/dp_plant.hpp"
#include "simulator.hpp"

int main(int argc, char *argv[], char *envp[]){

    std::string csv_file = ".";
    if (argc >= 1){csv_file = std::string(argv[1]);}


    //pendulum parameters
    double mass1 = 0.57288;
    double mass2 = 0.57288;
    double length1 = 0.5;
    double length2 = 0.5;
    double com1 = 0.5;
    double com2 = 0.5;
    double inertia1 = mass1*com1*com1;
    double inertia2 = mass2*com2*com2;
    double damping1 = 0.15;
    double damping2 = 0.15;
    double coulomb_friction1 = 0.0;
    double coulomb_friction2 = 0.0;
    double gravity = 9.81;
    double torque_limit1 = 10.0;
    double torque_limit2 = 10.0;


    double dt = 0.005;
    std::string integrator = "runge_kutta";

    DPPlant plant = DPPlant(true, true);
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

    CSVReader reader(csv_file, ",");
    std::vector<std::vector<double> > trajectory = reader.getDataDouble(1);
    int N = trajectory.size();
    std::cout << "N " << N << std::endl;
    std::cout << "len " << trajectory[0].size() << std::endl;

    //double pos_diff;
    Eigen::Vector<double, 4> s;
    for (int i=0; i<4; i++){
        s(i) = trajectory[0][1+i];
    }
    //s(0) = 1.5;
    //s(1) = 0.5;
    //s(2) = 0.2;
    //s(3) = 0.3;
    //Eigen::Vector<double, 4> s0 = {0., 0., 0., 0.};
    std::cout << "Start State (" << s(0) << ", " << s(1) << ", " << s(2) << ", " << s(3) << ")" << std::endl;

    Eigen::Vector<double, 2> u = {0., 0.};
    //sim.set_state(trajectory[0][0], trajectory[0][1], trajectory[0][2]);
    sim.set_state(0., s);
    for (int i=0; i<N; i++){
        for (int j=0; j<2; j++){
            u(j) = trajectory[i][5+j];
        }
        sim.step(u, dt, integrator);
        s = sim.get_state();
        std::cout << "State (" << s(0) << ", " << s(1) << ", " << s(2) << ", " << s(3) << ")";
        std::cout << "Action (" << u(0) << ", " << u(1) << ")" << std::endl;
        //pos_diff = sim.get_position() - trajectory[i+1][1];
        //if (pos_diff > 0.001){
        //    std::cout << trajectory[i][0] << ", " << sim.get_position() << ", " << trajectory[i+1][1] << ", " << pos_diff << std::endl;
        //}
    }

}
