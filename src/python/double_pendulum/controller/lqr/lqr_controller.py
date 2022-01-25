import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.lqr.lqr import lqr
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


class LQRController(AbstractController):
    def __init__(self,
                 mass=[0.5, 0.6],
                 length=[0.3, 0.2],
                 com=[0.3, 0.2],
                 damping=[0.1, 0.1],
                 coulomb_fric=[0.0, 0.0],
                 gravity=9.81,
                 inertia=[None, None],
                 torque_limit=[0.0, 1.0]):

        self.damping = np.asarray(damping)
        self.cfric = np.asarray(coulomb_fric)
        self.torque_limit = torque_limit

        self.splant = SymbolicDoublePendulum(mass=mass,
                                             length=length,
                                             com=com,
                                             damping=damping,
                                             gravity=gravity,
                                             coulomb_fric=coulomb_fric,
                                             inertia=inertia,
                                             torque_limit=torque_limit)

        # set default parameters
        self.set_goal()
        self.set_parameters()

    def set_goal(self, x=[np.pi, 0., 0., 0.]):
        self.xd = np.asarray(x)

    def set_parameters(self,
                       pp1_cost=1.,     # 1000., 0.001
                       pp2_cost=1.,     # 1000., 0.001
                       vv1_cost=1.,     # 1000.
                       vv2_cost=1.,     # 1000.
                       pv1_cost=0.,     # -500
                       pv2_cost=0.,     # -500
                       uu1_cost=0.01,  # 100., 0.01
                       uu2_cost=0.01):  # 100., 0.01
        # state cost matrix
        self.Q = np.array([[pp1_cost, pv1_cost, 0., 0.],
                           [pv1_cost, vv1_cost, 0., 0.],
                           [0., 0., pp2_cost, pv2_cost],
                           [0., 0., pv2_cost, vv2_cost]])

        # control cost matrix
        self.R = np.array([[uu1_cost, 0.], [0., uu2_cost]])
        # self.R = np.array([[uu_cost]])

    def init(self):
        Alin, Blin = self.splant.linear_matrices(x0=self.xd, u0=[0.0, 0.0])
        # Blin = Blin.T[1].T.reshape(4, 1)
        # print(Alin, Blin)
        self.K, self.S, _ = lqr(Alin, Blin, self.Q, self.R)

    def get_control_output(self, x):
        y = x.copy()
        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi

        y -= self.xd

        u = -self.K.dot(y)
        u = np.asarray(u)[0]
        # u = np.array([0., np.asarray(u)[0, 0]])

        # friction_compensation = (self.damping*x[2:]*x[2:]
        #                          + self.cfric*np.sign(x[2:]))
        # friction_compensation *= np.sign(self.torque_limit)
        # print("fric comp", friction_compensation, end="")
        # u += friction_compensation

        if y.dot(np.asarray(self.S.dot(y))[0]) > 15.0:  # old value:0.1
            u = [0., 0.]

        # print(x, u)
        return u
