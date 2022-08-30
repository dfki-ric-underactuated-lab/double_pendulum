import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.lqr.lqr import lqr
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.plant import DoublePendulumPlant


class LQRController(AbstractController):
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

        super().__init__()

        # self.damping = np.asarray(damping)
        # self.cfric = np.asarray(coulomb_fric)
        # self.torque_limit = torque_limit

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

        # set default parameters
        self.set_goal()
        self.set_parameters()
        self.set_cost_parameters()
        self.set_filter_args()

    def set_goal(self, x=[np.pi, 0., 0., 0.]):
        y = x.copy()
        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
        self.xd = np.asarray(y)

    def set_parameters(self, failure_value=np.nan,
                       cost_to_go_cut=15.):
        self.failure_value = failure_value
        self.cost_to_go_cut = cost_to_go_cut

    def set_cost_parameters(self,
                            p1p1_cost=1.,     # 1000., 0.001
                            p2p2_cost=1.,     # 1000., 0.001
                            v1v1_cost=1.,     # 1000.
                            v2v2_cost=1.,     # 1000.
                            p1p2_cost=0.,     # -500
                            v1v2_cost=0.,     # -500
                            p1v1_cost=0.,
                            p1v2_cost=0.,
                            p2v1_cost=0.,
                            p2v2_cost=0.,
                            u1u1_cost=0.01,    # 100., 0.01
                            u2u2_cost=0.01,    # 100., 0.01
                            u1u2_cost=0.):
        # state cost matrix
        self.Q = np.array([[p1p1_cost, p1p2_cost, p1v1_cost, p1v2_cost],
                           [p1p2_cost, p2p2_cost, p2v1_cost, p2v2_cost],
                           [p1v1_cost, p2v1_cost, v1v1_cost, v1v2_cost],
                           [p1v2_cost, p2v2_cost, v1v2_cost, v2v2_cost]])

        # control cost matrix
        self.R = np.array([[u1u1_cost, u1u2_cost], [u1u2_cost, u2u2_cost]])
        # self.R = np.array([[u2u2_cost]])

    # def set_cost_parameters_(self,
    #                          pars=[1., 1., 1., 1.,
    #                                0., 0., 0., 0., 0., 0.,
    #                                0.01, 0.01, 0.]):
    #     self.set_cost_parameters(p1p1_cost=pars[0],
    #                              p2p2_cost=pars[1],
    #                              v1v1_cost=pars[2],
    #                              v2v2_cost=pars[3],
    #                              p1v1_cost=pars[4],
    #                              p1v2_cost=pars[5],
    #                              p2v1_cost=pars[6],
    #                              p2v2_cost=pars[7],
    #                              u1u1_cost=pars[8],
    #                              u2u2_cost=pars[9],
    #                              u1u2_cost=pars[10])

    def set_cost_parameters_(self,
                             pars=[1., 1., 1., 1., 1.]):
        self.set_cost_parameters(p1p1_cost=pars[0],
                                 p2p2_cost=pars[1],
                                 v1v1_cost=pars[2],
                                 v2v2_cost=pars[3],
                                 p1v1_cost=0.0,
                                 p1v2_cost=0.0,
                                 p2v1_cost=0.0,
                                 p2v2_cost=0.0,
                                 u1u1_cost=pars[4],
                                 u2u2_cost=pars[4],
                                 u1u2_cost=0.0)

    def set_cost_matrices(self, Q, R):
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

    def init_(self):
        Alin, Blin = self.splant.linear_matrices(x0=self.xd, u0=[0.0, 0.0])
        self.K, self.S, _ = lqr(Alin, Blin, self.Q, self.R)

    def get_control_output_(self, x, t=None):
        y = x.copy()

        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi

        y -= self.xd

        u = -self.K.dot(y)
        u = np.asarray(u)[0]

        if y.dot(np.asarray(self.S.dot(y))[0]) > self.cost_to_go_cut:  # old value:0.1
            u = [self.failure_value, self.failure_value]

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        #print(x, u)
        return u


class LQRController_nonsymbolic(AbstractController):
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

        super().__init__()

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

        self.plant = DoublePendulumPlant(mass=self.mass,
                                         length=self.length,
                                         com=self.com,
                                         damping=self.damping,
                                         gravity=self.gravity,
                                         coulomb_fric=self.cfric,
                                         inertia=self.inertia,
                                         torque_limit=self.torque_limit)

        # set default parameters
        self.set_goal()
        self.set_parameters()
        self.set_cost_parameters()
        self.set_filter_args()

    def set_goal(self, x=[np.pi, 0., 0., 0.]):
        y = x.copy()
        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
        self.xd = np.asarray(y)

    def set_parameters(self, failure_value=np.nan,
                       cost_to_go_cut=15.):
        self.failure_value = failure_value
        self.cost_to_go_cut = cost_to_go_cut

    def set_cost_parameters(self,
                            p1p1_cost=1.,     # 1000., 0.001
                            p2p2_cost=1.,     # 1000., 0.001
                            v1v1_cost=1.,     # 1000.
                            v2v2_cost=1.,     # 1000.
                            p1p2_cost=0.,     # -500
                            v1v2_cost=0.,     # -500
                            p1v1_cost=0.,
                            p1v2_cost=0.,
                            p2v1_cost=0.,
                            p2v2_cost=0.,
                            u1u1_cost=0.01,    # 100., 0.01
                            u2u2_cost=0.01,    # 100., 0.01
                            u1u2_cost=0.):
        # state cost matrix
        self.Q = np.array([[p1p1_cost, p1p2_cost, p1v1_cost, p1v2_cost],
                           [p1p2_cost, p2p2_cost, p2v1_cost, p2v2_cost],
                           [p1v1_cost, p2v1_cost, v1v1_cost, v1v2_cost],
                           [p1v2_cost, p2v2_cost, v1v2_cost, v2v2_cost]])

        # control cost matrix
        self.R = np.array([[u1u1_cost, u1u2_cost], [u1u2_cost, u2u2_cost]])
        # self.R = np.array([[u2u2_cost]])

    # def set_cost_parameters_(self,
    #                          pars=[1., 1., 1., 1.,
    #                                0., 0., 0., 0., 0., 0.,
    #                                0.01, 0.01, 0.]):
    #     self.set_cost_parameters(p1p1_cost=pars[0],
    #                              p2p2_cost=pars[1],
    #                              v1v1_cost=pars[2],
    #                              v2v2_cost=pars[3],
    #                              p1v1_cost=pars[4],
    #                              p1v2_cost=pars[5],
    #                              p2v1_cost=pars[6],
    #                              p2v2_cost=pars[7],
    #                              u1u1_cost=pars[8],
    #                              u2u2_cost=pars[9],
    #                              u1u2_cost=pars[10])

    def set_cost_parameters_(self,
                             pars=[1., 1., 1., 1., 1.]):
        self.set_cost_parameters(p1p1_cost=pars[0],
                                 p2p2_cost=pars[1],
                                 v1v1_cost=pars[2],
                                 v2v2_cost=pars[3],
                                 p1v1_cost=0.0,
                                 p1v2_cost=0.0,
                                 p2v1_cost=0.0,
                                 p2v2_cost=0.0,
                                 u1u1_cost=pars[4],
                                 u2u2_cost=pars[4],
                                 u1u2_cost=0.0)

    def set_cost_matrices(self, Q, R):
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

    def init_(self):
        Alin, Blin = self.plant.linear_matrices(x0=self.xd, u0=[0.0, 0.0])
        self.K, self.S, _ = lqr(Alin, Blin, self.Q, self.R)

    def get_control_output_(self, x, t=None):
        y = x.copy()
        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi

        y -= self.xd

        u = -self.K.dot(y)
        u = np.asarray(u)[0]

        if y.dot(np.asarray(self.S.dot(y))[0]) > self.cost_to_go_cut:  # old value:0.1
            u = [self.failure_value, self.failure_value]

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        return u
