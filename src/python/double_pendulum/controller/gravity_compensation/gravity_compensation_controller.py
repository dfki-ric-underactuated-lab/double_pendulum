import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


class GravityCompensationController(AbstractController):
    def __init__(self,
                 mass=[0.5, 0.6],
                 length=[0.3, 0.2],
                 com=[0.3, 0.2],
                 damping=[0.1, 0.1],
                 coulomb_fric=[0.0, 0.0],
                 gravity=9.81,
                 inertia=[None, None],
                 torque_limit=[0.0, 1.0],
                 model_pars=None):

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.cfric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.cfric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            # self.Ir = model_pars.Ir
            # self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        self.splant = SymbolicDoublePendulum(mass=self.mass,
                                             length=self.length,
                                             com=self.com,
                                             damping=self.damping,
                                             gravity=self.gravity,
                                             coulomb_fric=self.cfric,
                                             inertia=self.inertia,
                                             torque_limit=self.torque_limit)

    def get_control_output(self, x, t=None):

        g = self.splant.gravity_vector(x)
        u = -np.dot(self.splant.B, g)

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        return u
