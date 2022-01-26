#include <stdlib.h>
#include <iostream>
#include <math.h>

#include "simulator.hpp"

const int sim_n_x = Simulator::n_x;

void Simulator::set_plant(DPPlant pl){
    plant = pl;
}

void Simulator::set_state(double t, Eigen::Vector<double, sim_n_x> x){
    time = t;
    for(int i=0; i<x.size(); i++){
        state(i) = x(i);
    }
}

Eigen::Vector<double, sim_n_x> Simulator::euler_integrator(
                                           double t,
                                           Eigen::Vector<double, sim_n_x> x,
                                           Eigen::Vector<double, n_u> u,
                                           double dt){
    Eigen::Vector<double, sim_n_x> xd = plant.rhs(t, x, u);
    return xd;
}

Eigen::Vector<double, sim_n_x> Simulator::runge_integrator(
                                           double t,
                                           Eigen::Vector<double, sim_n_x> x,
                                           Eigen::Vector<double, n_u> u,
                                           double dt){
    Eigen::Vector<double, sim_n_x> k1, k2, k3, k4;
    Eigen::Vector<double, sim_n_x> xd;

    k1 = plant.rhs(t, x, u);
    k2 = plant.rhs(t+0.5*dt, x+0.5*dt*k1, u);
    k3 = plant.rhs(t+0.5*dt, x+0.5*dt*k2, u);
    k4 = plant.rhs(t+dt, x+dt*k3, u);
    xd = (k1 + 2*(k2 + k3) + k4) / 6.0; 
    return xd;
}

void Simulator::step(Eigen::Vector<double, n_u> u,
                     double dt,
                     std::string integrator){
    double tl1 = plant.get_torque_limit1();
    double tl2 = plant.get_torque_limit2();
    if (u(0) > tl1){
        u(0) = tl1;
    }
    else if (u(0) < -tl1){
        u(0) = -tl1;
    }
    if (u(1) > tl2){
        u(1) = tl2;
    }
    else if (u(1) < -tl2){
        u(1) = -tl2;
    }

    Eigen::Vector<double, sim_n_x> xd;
    if (integrator == "euler"){
        xd = euler_integrator(time, state, u, dt);
    }
    else if (integrator == "runge_kutta"){
        xd = runge_integrator(time, state, u, dt);
    }
    state += dt*xd;
    time += dt;
}


void Simulator::simulate(double t0, double tf, double dt,
                         Eigen::Vector<double, sim_n_x> x,
                         Eigen::Vector<double, n_u> u,
                         std::string integrator){



    set_state(t0, x);
    while(time < tf){
        step(u, dt, integrator);
    }
}

