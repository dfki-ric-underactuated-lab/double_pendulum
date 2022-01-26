#include <stdlib.h>
#include <iostream>
#include <math.h>

#include "dp_plant.hpp"

const int n_x = DPPlant::n_x;
const int n_u = DPPlant::n_u;


DPPlant::DPPlant(){
    //default: both joints actuated
    B(0,0) = 1.;
    B(0,1) = 0.;
    B(1,0) = 0.;
    B(1,1) = 1.;
}

DPPlant::DPPlant(bool first_joint_actuated, bool second_joint_actuated){
    if (first_joint_actuated){
        B(0,0) = 1.;
    }
    else{
        B(0,0) = 0.;

    }
    B(0,1) = 0.;
    B(1,0) = 0.;
    if (second_joint_actuated){
        B(1,1) = 1.;
    }
    else{
        B(1,1) = 0.;

    }
}

//DPPlant::~DPPlant(){
//}

void DPPlant::set_parameters(double m1, double m2,
                             double l1, double l2,
                             double cm1, double cm2,
                             double I1, double I2,
                             double d1, double d2,
                             double cf1, double cf2,
                             double g,
                             double tl1, double tl2){
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


}


Eigen::Matrix<double, n_x/2, n_x/2> DPPlant::get_M(Eigen::Vector<double, n_x> x){

    Eigen::Matrix<double, n_x/2, n_x/2> M; // Mass matrix

    M(0,0) = inertia1 + inertia2 + mass2*pow(length1, 2.) + 2.*mass2*length1*com2*cos(x(1));
    M(0,1) = inertia2 + mass2*length1*com2*cos(x(1));
    M(1,0) = inertia2 + mass2*length1*com2*cos(x(1));
    M(1,1) = inertia2;

    if (verbose > 0){
        std::cout << "M:  ";
        for (int i=0; i<2; i++){
            for (int j=0; j<2; j++){
                std::cout << M(i,j) << " ";
            }
        }
        std::cout << std::endl;
    }
    return M;
}

Eigen::Matrix<double, n_x/2, n_x/2> DPPlant::get_Minv(Eigen::Vector<double, n_x> x){

    Eigen::Matrix<double, n_x/2, n_x/2> M = get_M(x);
    Eigen::Matrix<double, n_x/2, n_x/2> Minv; // Inverse mass matrix

    Minv = M.inverse();

    //double pre = 1. / (M(0,0)*M(1,1) - M(0,1)*M(1,0));
    //Minv(0,0) = pre*M(1,1);
    //Minv(0,1) = -pre*M(0,1);
    //Minv(1,0) = -pre*M(1,0);
    //Minv(1,1) = pre*M(0,0);

    if (verbose > 0){
        std::cout << "Minv:  ";
        for (int i=0; i<2; i++){
            for (int j=0; j<2; j++){
                std::cout << Minv(i,j) << " ";
            }
        }
        std::cout << std::endl;
    }
    return Minv;
}


//Eigen::Matrix<double, n_x/2, n_x/2> invert2x2(Eigen::Matrix<double, n_x/2, n_x/2> M){
//
//    Eigen::Matrix<double, n_x/2, n_x/2> Minv; // Inverse mass matrix
//
//    double pre = M(0,0)*M(1,1) - M(0,1)*M(1,0);
//
//    Minv(0,0) = pre*M(1,1);
//    Minv(0,1) = -pre*M(0,1);
//    Minv(1,0) = -pre*M(1,0);
//    Minv(1,1) = pre*M(0,0);
//
//    //std::cout << "Minv:  ";
//    //for (int i=0; i<2; i++){
//    //    for (int j=0; j<2; j++){
//    //        std::cout << Minv(i,j) << " ";
//    //    }
//    //}
//    //std::cout << std::endl;
//
//    return Minv;
//}


Eigen::Matrix<double, n_x/2, n_x/2> DPPlant::get_C(Eigen::Vector<double, n_x> x){

    Eigen::Matrix<double, 2, 2> C; // Coriolis

    C(0,0) = -2.*mass2*length1*com2*sin(x(1))*x(3);
    C(0,1) = -mass2*length1*com2*sin(x(1))*x(3);
    C(1,0) = mass2*length1*com2*sin(x(1))*x(2);
    C(1,1) = 0.;

    if (verbose > 0){
        std::cout << "C:  ";
        for (int i=0; i<2; i++){
            for (int j=0; j<2; j++){
                std::cout << C(i,j) << " ";
            }
        }
        std::cout << std::endl;
    }
    return C;
}


Eigen::Vector<double, n_x/2> DPPlant::get_G(Eigen::Vector<double, n_x> x){

    Eigen::Vector<double, 2> G; // Gravity vctror

    G(0) = -mass1*gravity*com1*sin(x(0)) 
             - mass2*gravity*(length1*sin(x(0)) + com2*sin(x(0)+x(1)));
    G(1) = -mass2*gravity*com2*sin(x(0)+x(1));

    if (verbose > 0){
        std::cout << "G:  ";
        for (int i=0; i<2; i++){
            std::cout << G(i) << " ";
        }
        std::cout << std::endl;
    }
    return G;
}

Eigen::Vector<double, n_x/2> DPPlant::get_F(Eigen::Vector<double, n_x> x){

    Eigen::Vector<double, n_x/2> F; // Coulomb vector

    double sign_v1 = (double) (x(2) > 0) - (x(2) < 0);
    double sign_v2 = (double) (x(3) > 0) - (x(3) < 0);

    F(0) = damping1*x(2) + coulomb_friction1*sign_v1;
    F(1) = damping2*x(3) + coulomb_friction2*sign_v2;

    ////std::cout << "F inputs: " << damping1 << " " << damping2 << " " << coulomb_friction1 << " " << coulomb_friction2 << " " << sign_v1 << " " << sign_v2 << std::endl;
    if (verbose > 0){
        std::cout << "F:  ";
        for (int i=0; i<2; i++){
            std::cout << F(i) << " ";
        }
        std::cout << std::endl;
    }
    return F;
}


Eigen::Vector<double, n_x/2> DPPlant::forward_dynamics(Eigen::Vector<double, n_x> x,
                                                       Eigen::Vector<double, n_u> u){

    Eigen::Vector<double, n_x/2> vel;
    vel(0) = x(2);
    vel(1) = x(3);
    Eigen::Vector<double, n_x/2> acc;

    //Eigen::Matrix<double, n_x/2, n_x/2> M;
    Eigen::Matrix<double, n_x/2, n_x/2> Minv;
    Eigen::Matrix<double, n_x/2, n_x/2> C;
    Eigen::Vector<double, n_x/2> G; // Gravity vctror
    Eigen::Vector<double, n_x/2> F; // Coulomb vector

    //M = get_M(x);
    //Minv = invert2x2(M);
    Minv = get_Minv(x);
    C = get_C(x);
    G = get_G(x);
    F = get_F(x);

    acc = Minv*(G + B*u - C*vel - F);

    //Eigen::Matrix<double, n_x/2, 1> accn;
    ////accn.pos1 = s.vel1;
    ////accn.pos1 = s.vel2;
    //accn(0) = acc(0,0);
    //accn(1) = acc(1,0);

    return acc;
}


Eigen::Vector<double, n_x/2> DPPlant::forward_dynamics(double pos1, double pos2,
                                                       double vel1, double vel2,
                                                       double u1, double u2){

    Eigen::Vector<double, n_x> x;
    x(0) = pos1;
    x(1) = pos2;
    x(2) = vel1;
    x(3) = vel2;

    Eigen::Vector<double, n_u> u;
    u(0) = u1;
    u(1) = u2;

    return forward_dynamics(x, u);
}

Eigen::Vector<double, n_x> DPPlant::rhs(double t, double pos1, double pos2,
                                        double vel1, double vel2,
                                        double u1, double u2){

    Eigen::Vector<double, n_x/2> accn = forward_dynamics(pos1, pos2,
                                                         vel1, vel2,
                                                         u1, u2);
    //std::cout << "vel:  ";
    //std::cout << vel1 << " ";
    //std::cout << vel2 << " ";
    //std::cout << std::endl;

    //std::cout << "accn:  ";
    //std::cout << accn(0) << " ";
    //std::cout << accn(1) << " ";
    //std::cout << std::endl;
 
    Eigen::Vector<double, n_x> xd;

    xd(0) = vel1;
    xd(1) = vel2;
    xd(2) = accn(0);
    xd(3) = accn(1);
    return xd;
}

Eigen::Vector<double, n_x> DPPlant::rhs(double t,
                                        Eigen::Vector<double, n_x> x,
                                        Eigen::Vector<double, n_u> u){

    Eigen::Vector<double, n_x/2> accn = forward_dynamics(x, u);
    Eigen::Vector<double, n_x> xd;

    xd(0) = x(2);
    xd(1) = x(3);
    xd(2) = accn(0);
    xd(3) = accn(1);
    return xd;
}


double DPPlant::calculate_potential_energy(Eigen::Vector<double, n_x> x){
    // 0 level at hinge
    double y1, y2, pot;
    y1 = -com1*cos(x(0));
    y2 = -length1*cos(x(0)) - com2*cos(x(0) + x(1));
    pot = mass1*gravity*y1 + mass2*gravity*y2;
    return pot;
}

double DPPlant::calculate_kinetic_energy(Eigen::Vector<double, n_x> x){
    Eigen::Matrix<double, 2, 1> vel;
    vel(0,0) = x(2);
    vel(1,0) = x(3);
    Eigen::Matrix<double, n_x/2, n_x/2> M = get_M(x);
    double kin =  vel.transpose()*M*vel;
    return kin;
}

double DPPlant::calculate_total_energy(Eigen::Vector<double, n_x> x){
    return calculate_potential_energy(x) + calculate_kinetic_energy(x);
}
Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> DPPlant::get_Mx(Eigen::Vector<double, n_x> x){

    //Eigen::Tensor<double, 3> Mx;
    Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> Mx;

    Mx(0)(0,0) = 0.;
    Mx(0)(0,1) = 0.;
    Mx(0)(1,0) = 0.;
    Mx(0)(1,1) = 0.;

    Mx(1)(0,0) = -2.*length1*mass2*com2*sin(x(1));
    Mx(1)(0,1) = -length1*mass2*com2*sin(x(1));
    Mx(1)(1,0) = -length1*mass2*com2*sin(x(1));
    Mx(1)(1,1) = 0.;

    Mx(2)(0,0) = 0.;
    Mx(2)(0,1) = 0.;
    Mx(2)(1,0) = 0.;
    Mx(2)(1,1) = 0.;

    Mx(3)(0,0) = 0.;
    Mx(3)(0,1) = 0.;
    Mx(3)(1,0) = 0.;
    Mx(3)(1,1) = 0.;

    return Mx;
}

Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> DPPlant::get_Minvx(Eigen::Vector<double, n_x> x){

    Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> Minvx;

    Minvx(0)(0,0) = 0.;
    Minvx(0)(0,1) = 0.;
    Minvx(0)(1,0) = 0.;
    Minvx(0)(1,1) = 0.;

    double den = -inertia1*inertia2 - inertia2*pow(length1, 2.)*mass2 + pow(length1*mass2*com2*cos(x(1)), 2.);

    Minvx(1)(0,0) = -2.*inertia2*pow((length1*mass2*com2), 2.)*sin(x(1))*cos(x(1))/pow(den, 2.);
    Minvx(1)(0,1) = 2*pow(length1*mass2*com2, 2.)*(inertia2 + length1*mass2*com2*cos(x(1)))*sin(x(1))*cos(x(1)) / pow(den, 2.)
                   - length1*mass2*com2*sin(x(1)) / den;
    Minvx(1)(1,0) = 2*pow(length1*mass2*com2, 2.)*(inertia2 + length1*mass2*com2*cos(x(1)))*sin(x(1))*cos(x(1)) / pow(den, 2.)
                   - length1*mass2*com2*sin(x(1)) / den;
    Minvx(1)(1,1) = 2*pow(length1*mass2*com2, 2.)*(-inertia1-inertia2-pow(length1, 2.)*mass2 - 2*length1*mass2*com2*cos(x(1)))*sin(x(1))*cos(x(1)) / pow(den, 2.)
                   + 2*length1*mass2*com2*sin(x(1)) / den;

    Minvx(2)(0,0) = 0.;
    Minvx(2)(0,1) = 0.;
    Minvx(2)(1,0) = 0.;
    Minvx(2)(1,1) = 0.;

    Minvx(3)(0,0) = 0.;
    Minvx(3)(0,1) = 0.;
    Minvx(3)(1,0) = 0.;
    Minvx(3)(1,1) = 0.;

    return Minvx;
}

Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> DPPlant::get_Cx(Eigen::Vector<double, n_x> x){

    Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> Cx;

    Cx(0)(0,0) = 0.;
    Cx(0)(0,1) = 0.;
    Cx(0)(1,0) = 0.;
    Cx(0)(1,1) = 0.;

    Cx(1)(0,0) = -2.*length1*mass2*x(3)*com2*cos(x(1));
    Cx(1)(0,1) = -length1*mass2*x(3)*com2*cos(x(1));
    Cx(1)(1,0) = length1*mass2*x(2)*com2*cos(x(1));
    Cx(1)(1,1) = 0.;

    Cx(2)(0,0) = 0.;
    Cx(2)(0,1) = 0.;
    Cx(2)(1,0) = length1*mass2*com2*sin(x(1));
    Cx(2)(1,1) = 0.;

    Cx(3)(0,0) = -2*length1*mass2*com2*sin(x(1));
    Cx(3)(0,1) = -length1*mass2*com2*sin(x(1));
    Cx(3)(1,0) = 0.;
    Cx(3)(1,1) = 0.;

    return Cx;
}


Eigen::Matrix<double, n_x/2, n_x> DPPlant::get_Gx(Eigen::Vector<double, n_x> x){

    Eigen::Matrix<double, n_x/2, n_x> Gx;

    Gx(0,0) = -gravity*mass1*com1*cos(x(0))
              -gravity*mass2*(length1*cos(x(0)) + com2*cos(x(0)+x(1)));
    Gx(0,1) = -gravity*mass2*com2*cos(x(0)+x(1));
    Gx(0,2) = 0.;
    Gx(0,3) = 0.;

    Gx(1,0) = -gravity*mass2*com2*cos(x(0)+x(1));
    Gx(1,1) = -gravity*mass2*com2*cos(x(0)+x(1));
    Gx(1,2) = 0.;
    Gx(1,3) = 0.;

    return Gx;
}

Eigen::Matrix<double, n_x/2, n_x> DPPlant::get_Fx(Eigen::Vector<double, n_x> x){

    Eigen::Matrix<double, n_x/2, n_x> Fx;

    Fx(0,0) = 0.;
    Fx(0,1) = 0.;
    Fx(0,2) = damping1;
    Fx(0,3) = 0.;

    Fx(1,0) = 0.;
    Fx(1,1) = 0.;
    Fx(1,2) = 0.;
    Fx(1,3) = damping2;

    return Fx;
}

Eigen::Matrix<double, n_x, n_x> DPPlant::get_dynx(Eigen::Vector<double, n_x> x,
                                                  Eigen::Vector<double, n_u> u){

    Eigen::Matrix<double, n_x, n_x> dynx;
    Eigen::Matrix<double, n_x/2, n_x> dynx_upper, dynx_lower;

    Eigen::Vector<double, 2> vel;
    vel(0) = x(2);
    vel(1) = x(3);

    Eigen::Vector<double, n_x/2> eom, tmp2a, tmp2b;
    Eigen::Matrix<double, n_x/2, n_x> tmp24a, tmp24b, tmp24c;

    // dynamics matrices and derivatives
    Eigen::Matrix<double, n_x/2, n_x/2> M, Minv, C;
    Eigen::Vector<double, 2> G, F;
    Eigen::Vector<Eigen::Matrix<double, n_x/2, n_x/2>, n_x> Minvx, Cx;
    Eigen::Matrix<double, n_x/2, n_x> Gx, Fx;
    //M = get_M(x);
    Minv = get_Minv(x);
    C = get_C(x);
    G = get_G(x);
    F = get_F(x);
    Minvx = get_Minvx(x);
    Cx = get_Cx(x);
    Gx = get_Gx(x);
    Fx = get_Fx(x);

    // dynx_upper = del(qd)/del(x)
    // dynx_lower = del(qdd)/del(x)
    dynx_upper(0,0) = 0.;
    dynx_upper(0,1) = 0.;
    dynx_upper(0,2) = 1.;
    dynx_upper(0,3) = 0.;
    dynx_upper(1,0) = 0.;
    dynx_upper(1,1) = 0.;
    dynx_upper(1,2) = 0.;
    dynx_upper(1,3) = 1.;

    eom = G + B*u - C*vel - F;

    for (int i=0; i<n_x; i++){
        tmp2a = Minvx(i)*eom;
        tmp24a(0,i) = tmp2a(0);
        tmp24a(1,i) = tmp2a(1);

        for (int j=0; j<n_x; j++){
            tmp2b = Cx(i)*vel;
            tmp24b(0,i) = tmp2b(0);
            tmp24b(1,i) = tmp2b(1);
        }
        

        tmp24c = Minv*(-tmp24b - C*dynx_upper + Gx - Fx);
    }

    dynx_lower = tmp24a + tmp24c;

    for (int j=0; j<n_x/2; j++){
        for (int i=0; i<n_x; i++){
            dynx(j, i) = dynx_upper(j, i);
            dynx(n_x/2+j, i) = dynx_lower(j, i);
        }
    }

    return dynx;
}

Eigen::Matrix<double, n_x, n_u> DPPlant::get_dynu(Eigen::Vector<double, n_x> x,
                                                  Eigen::Vector<double, n_u> u){

    Eigen::Matrix<double, n_x, n_u> dynu;
    Eigen::Matrix<double, n_x/2, n_u> dynu_lower;
    Eigen::Matrix<double, n_x/2, n_x/2> Minv = get_Minv(x);

    dynu_lower = Minv*B;

    for (int i=0; i<n_x/2; i++){
        for (int j=0; j<n_u; j++){
            dynu(i,j) = 0.;
            dynu(n_x/2+i,j) = dynu_lower(i, j);
        }
    }

    return dynu;
}

Eigen::Vector<double, n_x> DPPlant::get_Ex(Eigen::Vector<double, n_x> x){

    Eigen::Vector<double, n_x> E_x;
    E_x(0) = gravity*mass1*com1*sin(x(0)) + gravity*mass2*(-length1*sin(x(0)) + com2*sin(x(0)+x(1)));
    E_x(1) = gravity*mass2*com2*sin(x(0)+x(1)) -
             0.5*length1*mass2*x(2)*x(3)*com2*sin(x(1)) +
             0.5*x(2)*(-2.*length1*mass2*x(2)*com2*sin(x(1) - length1*mass2*x(3)*com2*sin(x(1))));
    E_x(2) = x(2)*(inertia1+inertia2+pow(length2, 2.)*mass2+2.*length1*mass2*com2*cos(x(1))) +
             x(3)*(inertia2+length1*mass2*com2*cos(x(1)));
    E_x(3) = inertia2*x(3) + x(2)*(inertia2*length1*mass2*com2*cos(x(1)));

    return E_x;
}

Eigen::Matrix<double, n_x, n_x> DPPlant::get_Exx(Eigen::Vector<double, n_x> x){

    Eigen::Matrix<double, n_x, n_x> E_xx;

    E_xx(0,0) = gravity*mass1*com1*cos(x(0)) + gravity*mass2*(-length1*cos(x(0)) + com2*cos(x(0)+x(1)));
    E_xx(0,1) = gravity*mass2*com2*cos(x(0)+x(1));
    E_xx(0,2) = 0.;
    E_xx(0,3) = 0.;

    E_xx(1,0) = gravity*mass2*com2*cos(x(0)+x(1));
    E_xx(1,1) = gravity*mass2*com2*cos(x(0)*x(1)) - 
                0.5*length1*mass2*x(2)*x(3)*com2*cos(x(1)) + 
                0.5*x(2)*(-2.*length1*mass2*x(2)*com2*cos(x(1)) - length1*mass2*x(3)*com2*cos(x(1)));
    E_xx(1,2) = -2.*length1*mass2*x(2)*com2*sin(x(1)) - length1*mass2*x(3)*com2*sin(x(1));
    E_xx(1,3) = -length1*mass2*x(2)*com2*sin(x(1));

    E_xx(2,0) = 0.;
    E_xx(2,1) = -2.*length1*mass2*x(2)*com2*sin(x(1)) - length1*mass2*x(3)*com2*sin(x(1));
    E_xx(2,2) = inertia1 + inertia2 + pow(length1, 2.)*mass2 + 2.*length1*mass2*com2*cos(x(1));
    E_xx(2,3) = inertia2 + length1*mass2*com2*cos(x(1));

    E_xx(3,0) = 0.;
    E_xx(3,1) = -length1*mass2*x(2)*com2*sin(x(1));
    E_xx(3,2) = inertia2 + length1*mass2*com2*cos(x(1));
    E_xx(3,3) = inertia2;

    return E_xx;
}
