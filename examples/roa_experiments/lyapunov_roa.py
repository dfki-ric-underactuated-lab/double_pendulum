import numpy as np
import matplotlib.pyplot as plt

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.lqr.lqr_controller import LQRController


class Lyapunov_function():
    def __init__(self,
                 mass,
                 length,
                 gravity,
                 torque_limit,
                 fixpoint):

        self.plant = SymbolicDoublePendulum(mass=mass,
                                            length=length,
                                            com=length,
                                            damping=[0., 0.],
                                            gravity=gravity,
                                            coulomb_fric=[0., 0.],
                                            inertia=[mass[0]*length[0]**2., mass[1]*length[1]**2.],
                                            torque_limit=torque_limit)

        self.controller = LQRController(mass=mass,
                                        length=length,
                                        com=length,
                                        damping=[0., 0.],
                                        gravity=gravity,
                                        coulomb_fric=[0., 0.],
                                        inertia=[mass[0]*length[0]**2., mass[1]*length[1]**2.],
                                        torque_limit=torque_limit)
        self.controller.set_goal(fixpoint)
        # self.controller.set_cost_parameters(p1p1_cost=16.50,
        #                                     p2p2_cost=90.94,
        #                                     v1v1_cost=0.07,
        #                                     v2v2_cost=0.01,
        #                                     p1v1_cost=0.,
        #                                     p1v2_cost=0.,
        #                                     p2v1_cost=0.,
        #                                     p2v2_cost=0.,
        #                                     u1u1_cost=3.65,
        #                                     u2u2_cost=3.65,
        #                                     u1u2_cost=0.)
        self.controller.set_cost_parameters(p1p1_cost=10.,
                                            p2p2_cost=10.,
                                            v1v1_cost=1.,
                                            v2v2_cost=1.,
                                            p1v1_cost=0.,
                                            p1v2_cost=0.,
                                            p2v1_cost=0.,
                                            p2v2_cost=0.,
                                            u1u1_cost=1.,
                                            u2u2_cost=1.,
                                            u1u2_cost=0.)
        self.controller.set_parameters(failure_value=0.0)
        self.controller.init()

        self.S = np.asarray(self.controller.S)

        self.fixpoint = fixpoint

    def get_V(self, x):
        x_error = x - self.fixpoint
        V = np.dot(x_error, np.dot(self.S, x_error))
        return V

    def get_Vdot(self, x):
        x_error = x - self.fixpoint
        u = self.controller.get_control_output(x)
        xd = self.plant.rhs(0., x, u)
        Vdot = 2.*np.dot(x_error, np.dot(self.S, xd))
        return Vdot


mass = [0.608, 0.630]
length = [0.3, 0.2]
gravity = 9.81
torque_limit = [0.0, 4.0]

N = 10000
fixpoint = np.array([np.pi, 0., 0., 0.])
sample_range = np.array([0.1, 0.1, 0.1, 0.1])

Lf = Lyapunov_function(mass,
                       length,
                       gravity,
                       torque_limit,
                       fixpoint)

c_star = 1e9

c_list = []

for i in range(N):
    s = np.random.uniform(low=-sample_range, high=sample_range)
    s += fixpoint
    v = Lf.get_V(s)
    vdot = Lf.get_Vdot(s)

    # print(s, v, vdot)
    if (vdot >= 0. and v < c_star):
        #c_star = np.copy(v)
        c_star = v
    c_list.append(c_star)

print(c_star)

plt.plot(np.arange(len(c_list)), c_list)
plt.show()
