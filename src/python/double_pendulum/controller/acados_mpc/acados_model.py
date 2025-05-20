import os
from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_solver import AcadosOcpSolver
from acados_template.acados_sim_solver import AcadosSimSolver
import casadi as cas
import numpy as np
import pandas as pd
from double_pendulum.controller.abstract_controller import AbstractController
import yaml

class PendulumModel:

    model = AcadosModel()

    th1 = cas.SX.sym("th1")
    th2 = cas.SX.sym("th2")
    thd1 = cas.SX.sym("thd1")
    thd2 = cas.SX.sym("thd2")

    th = cas.vertcat(th1, th2)
    thd = cas.vertcat(thd1, thd2)

    # shape(4,)
    x = cas.vertcat(th, thd)
    xdot = cas.SX.sym("xdot", x.shape)

    # Torque Input
    u1 = cas.SX.sym("u1")
    u2 = cas.SX.sym("u2")

    # shape(2,)
    u = cas.vertcat(u1, u2)

    x_labels = [
        r"$\theta$1 [rad]",
        r"$\theta$2 [rad]",
        r"$\dot{\theta}1$ [rad/s]",
        r"$\dot{\theta}2$ [rad/s]",
    ]
    u_labels = ["u1", "$F$2"]
    t_label = "$t$ [s]"

    def __init__(
        self,
        mass=[1.0, 1.0],
        length=[0.5, 0.5],
        inertia=[None, None],
        center_of_mass=[0.5, 0.5],
        damping=[0.5, 0.5],
        coulomb_fric=[0.0, 0.0],
        gravity=9.81,
        motor_intertia=0.0,
        gear_ratio=6,
        actuated_joint=0,
        p_global=False,
        nonlinear_params=False
    ):

        self.m1 = mass[0]  # mass of the ball [kg]
        self.m2 = mass[1]  # mass of the ball [kg]
        self.l1 = length[0]  # length of the rod [m]
        self.l2 = length[1]  # length of the rod [m]
        self.I1 = inertia[0] if inertia[0] else self.m1 * self.l1**2  # inertia
        self.I2 = inertia[1] if inertia[1] else self.m2 * self.l2**2  # inertia
        self.r1 = center_of_mass[0]
        self.r2 = center_of_mass[1]
        self.g = gravity  # gravity constant [m/s^2]
        self.Ir = motor_intertia  # motor inertia
        self.gr = gear_ratio  # gear ratio
        self.b1 = damping[0]
        self.b2 = damping[1]
        self.cf1 = coulomb_fric[0]
        self.cf2 = coulomb_fric[1]

        if actuated_joint == 0:
            self.B = cas.SX([[1, 0], [0, 0]])
        elif actuated_joint == 1:
            self.B = cas.SX([[0, 0], [0, 1]])
        else:
            self.B = cas.SX([[1, 0], [0, 1]])

        # mass matrix
        # shape (2,2)
        self.M = cas.blockcat(
            self.I1
            + self.I2
            + self.m2 * self.l1**2
            + 2 * self.m2 * self.l1 * self.r2 * cas.cos(self.th2)
            + self.gr**2.0 * self.Ir
            + self.Ir,
            self.I2
            + self.m2 * self.l1 * self.r2 * cas.cos(self.th2)
            - self.gr * self.Ir,
            self.I2
            + self.m2 * self.l1 * self.r2 * cas.cos(self.th2)
            - self.gr * self.Ir,
            self.I2 + self.gr**2.0 * self.Ir,
        )

        self.inv_M = cas.solve(self.M, cas.SX.eye(self.M.size1()))

        # Coriolis Matrix
        # shape (2,2)
        self.C = cas.blockcat(
            -2 * self.m2 * self.l1 * self.r2 * cas.sin(self.th2) * self.thd2,
            -self.m2 * self.l1 * self.r2 * cas.sin(self.th2) * self.thd2,
            self.m2 * self.l1 * self.r2 * cas.sin(self.th2) * self.thd1,
            0,
        )

        # self.gravity matrix
        # shape (2,1)
        self.tau = cas.vertcat(
            -self.m1 * self.g * self.r1 * cas.sin(self.th1)
            - self.m2
            * self.g
            * (self.l1 * cas.sin(self.th1) + self.r2 * cas.sin(self.th1 + self.th2)),
            -self.m2 * self.g * self.r2 * cas.sin(self.th1 + self.th2),
        )

        # Coulomb Vector
        # shape (2,1)
        if p_global or nonlinear_params:
            steepness = cas.SX.sym("steepness")
        else:
            steepness=100

        self.F = cas.vertcat(
            self.b1 * self.thd1 + self.cf1 * cas.atan(steepness * self.thd1),
            self.b2 * self.thd2 + self.cf2 * cas.atan(steepness * self.thd2),
        )

        if (
            self.cf1 <= 0.0001 and self.cf2 <= 0.0001 and self.b1 <= 0.0001 and
            self.b2 <= 0.0001
        ):
            print("Friction parameters are zero, removing friction term from model")
            self.F = cas.SX([0, 0])

        self.eom = self.inv_M @ (
            self.B @ self.u - self.C @ self.thd + self.tau - self.F
        )
        f_expl = cas.vertcat(self.thd, self.eom)
        f_impl = self.xdot - f_expl

        self.model.f_impl_expr = f_impl
        self.model.f_expl_expr = f_expl
        self.model.x = self.x
        self.model.xdot = self.xdot
        self.model.u = self.u
        self.model.name = "double_pendulum"

        if p_global:
            self.model.p_global = steepness
        if nonlinear_params:
            self.model.p = steepness

        K1 = 1 / 2 * self.I1 * self.thd1**2
        K2 = (
            1
            / 2
            * (
                self.m2 * self.l1**2
                + self.I2
                + 2 * self.m2 * self.l1 * self.r2 * cas.cos(self.th2)
            )
            * self.th1**2
            + 1 / 2 * self.I2 * self.thd2**2
            + (self.I2 + self.m2 * self.l1 * self.r2 * cas.cos(self.th2))
            * self.thd1
            * self.thd2
        )
        K = K1 + K2

        P1 = -self.m1 * self.g * self.r1 * cas.cos(self.th1)
        P2 = (
            -self.m2
            * self.g
            * (self.l1 * cas.cos(self.th1) + self.r2 * cas.cos(self.th1 + self.th2))
        )
        P = P1 + P2

        self.model.P1 = P1
        self.model.P2 = P2
        self.model.K1 = K1
        self.model.K2 = K2

        self.K1 = K1
        self.K2 = K2
        self.P1 = P1
        self.P2 = P2

        # for later use with own integrator

    def forward_dynamics(self, x, u):
        bu = self.B @ self.u
        rest = (self.C @ self.thd) + self.tau - self.F
        fun = cas.Function("fun", [self.x, self.u], [self.eom])
        force = cas.Function("Bu", [self.x, self.u], [bu])
        minv = cas.Function("rest", [self.x, self.u], [rest])
        acc = fun(x, u)
        return acc  #

    def rhs(self, t, x, u):
        accn = self.forward_dynamics(x, u)

        # Next state
        res = np.zeros(2 * 2)
        res[0] = x[2]
        res[1] = x[3]
        res[2] = accn[0]
        res[3] = accn[1]

        return res

    # for better verbosity
    def mass_matrix(self, x):
        fun = cas.Function("mass_matrix", [self.x], [self.M])
        m = fun(x)
        return m

    def coriolis_matrix(self, x):
        fun = cas.Function("coriolis_matrix", [self.x], [self.C])
        c = fun(x)
        return c

    def gravity_matrix(self, x):
        fun = cas.Function("gravity_matrix", [self.x], [self.tau])
        t = fun(x)
        return t

    def coulomb_vector(self, x):
        fun = cas.Function("coulomb_vector", [self.x], [self.F])
        f = fun(x)
        return f

    def kinetic_energy(self, x):
        getK1 = cas.Function("kinetic_energy1", [self.x], [self.K1])
        k1 = getK1(x)

        getK2 = cas.Function("kinetic_energy2", [self.x], [self.K2])
        k2 = getK2(x)

        return k1, k2

    def potential_energy(self, x):
        getP1 = cas.Function("potential_energy1", [self.x], [self.P1])
        p1 = getP1(x)

        getP2 = cas.Function("potential_energy2", [self.x], [self.P2])
        p2 = getP2(x)

        return p1, p2

    def acados_model(self):
        return self.model
