#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <string>
#include <Eigen/Dense>

#include "../../model/src/dp_plant.hpp"

class Simulator{

private:

    DPPlant plant;

public:
    static const int n_x = DPPlant::n_x;
    static const int n_u = DPPlant::n_u;

    Eigen::Vector<double, n_x> state;
    double time;

    Eigen::Vector<double, n_x> euler_integrator(double,
                                                Eigen::Vector<double, n_x>,
                                                Eigen::Vector<double, n_u>,
                                                double);
    Eigen::Vector<double, n_x> runge_integrator(double,
                                                Eigen::Vector<double, n_x>,
                                                Eigen::Vector<double, n_u>,
                                                double);

    void set_plant(DPPlant);
    Eigen::Vector<double, n_x> get_state() {return state;};

    void set_state(double, Eigen::Vector<double, n_x>);
    void step(Eigen::Vector<double, n_u>, double, std::string);
    void simulate(double, double, double,
                  Eigen::Vector<double, n_x>,
                  Eigen::Vector<double, n_u>,
                  std::string);

};

#endif // SIMULATOR_HPP

