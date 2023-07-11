#ifndef DP_PLANT2_HPP
#define DP_PLANT2_HPP

#include <string>
#include <Eigen/Dense>
//#include <vector>

namespace Eigen {
    template<typename Type, int sz>
    using Vector = Matrix<Type, sz, 1>;
}

class DPPlant{

    double mass1, mass2;
    double length1, length2;
    double com1, com2;
    double inertia1, inertia2;
    double motor_inertia;
    double gear_ratio;
    double damping1, damping2;
    double coulomb_friction1, coulomb_friction2;
    double gravity;
    double torque_limit1, torque_limit2;

public:

    static const int n_x = 4;
    static const int n_u = 2;

    int verbose = 0;

    Eigen::Matrix<double, n_x/2, n_u> B; // Actuator selection


    DPPlant();
    DPPlant(bool, bool);
    //~DPPlant();

    void set_parameters(double, double, double, double, double,
                        double, double, double, double, double,
                        double, double, double, double, double);
    double get_torque_limit1() {return torque_limit1;};
    double get_torque_limit2() {return torque_limit2;};

    Eigen::Matrix<double, n_x/2, n_x/2> get_M(Eigen::Vector<double, n_x>);
    Eigen::Matrix<double, n_x/2, n_x/2> get_Minv(Eigen::Vector<double, n_x>);
    Eigen::Matrix<double, n_x/2, n_x/2> get_C(Eigen::Vector<double, n_x>);
    Eigen::Vector<double, n_x/2> get_G(Eigen::Vector<double, n_x>);
    Eigen::Vector<double, n_x/2> get_F(Eigen::Vector<double, n_x>);
    Eigen::Matrix<double, n_x/2, n_u> get_B() {return B;};

    //Eigen::Matrix<double, n_x/2, n_x/2> invert2x2(Eigen::Matrix<double, n_x/2, n_x/2>);

    Eigen::Vector<double, n_x/2> forward_dynamics(double, double, double, double, double,
                            double);
    Eigen::Vector<double, n_x/2> forward_dynamics(Eigen::Vector<double, n_x>, Eigen::Vector<double, n_u>);
    Eigen::Vector<double, n_x> rhs(double, double, double, double, double, double,
                            double);
    Eigen::Vector<double, n_x> rhs(double, Eigen::Vector<double, n_x> , Eigen::Vector<double, n_u>);

    double calculate_potential_energy(Eigen::Vector<double, n_x>);
    double calculate_kinetic_energy(Eigen::Vector<double, n_x>);
    double calculate_total_energy(Eigen::Vector<double, n_x>);

    Eigen::Vector<double, n_x> get_Ex(Eigen::Vector<double, n_x>);
    Eigen::Matrix<double, n_x, n_x> get_Exx(Eigen::Vector<double, n_x>);

    Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> get_Mx(Eigen::Vector<double, n_x>);
    Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> get_Minvx(Eigen::Vector<double, n_x>);
    Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> get_Cx(Eigen::Vector<double, n_x>);
    Eigen::Matrix<double, n_x/2, n_x> get_Gx(Eigen::Vector<double, n_x>);
    Eigen::Matrix<double, n_x/2, n_x> get_Fx(Eigen::Vector<double, n_x>);

    Eigen::Matrix<double, n_x, n_x> get_dynx(Eigen::Vector<double, n_x>,
                                             Eigen::Vector<double, n_u>);
    Eigen::Matrix<double, n_x, n_u> get_dynu(Eigen::Vector<double, n_x>,
                                             Eigen::Vector<double, n_u>);

};

#endif // DP_PLANT2_HPP
