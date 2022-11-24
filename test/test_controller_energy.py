"""
Unit Tests
==========
"""

import unittest
import numpy as np


from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.energy.energy_Xin import EnergyController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator


class Test(unittest.TestCase):


    mpar = model_parameters()
    torque_limits = [[0., 25.]]

    goals = [[np.pi, 0., 0., 0.]]

    states =[[0., 0., 0., 0.],
             [np.pi, 0., 0., 0.],
             [0., 1.0, -5, 12.],
             [5*np.pi, -3*np.pi, -1e-5, 100],
             [1e5, 1e-11, 1, -3],
             [1, 0, 0, 1],
            ]

    mass = [1.0, 1.0]
    length = [1.0, 2.0]
    com = [0.5, 1.0]
    damping = [0.0, 0.0]
    cfric = [0.0, 0.0]
    gravity = 9.8
    inertia = [mass[0]*com[0]**2+0.083, mass[1]*com[1]**2+0.33]
    motor_inertia = 0.
    torque_limit = [0.0, 25.0]

    integrator = "euler"
    goal = [np.pi, 0., 0., 0.]
    dt = 0.0045
    x0 = [np.pi/2.-1.4, 0.0, 0.0, 0.0]
    t_final = 30.0

    kp = 61.2  # > 61.141
    kd = 35.8  # > 35.741
    kv = 66.3  # > 0.0

    plant = SymbolicDoublePendulum(mass=mass,
                                   length=length,
                                   com=com,
                                   damping=damping,
                                   gravity=gravity,
                                   coulomb_fric=cfric,
                                   inertia=inertia,
                                   motor_inertia=motor_inertia,
                                   torque_limit=torque_limit)
    sim = Simulator(plant=plant)

    controller = EnergyController(mass=mass,
                                  length=length,
                                  com=com,
                                  damping=damping,
                                  gravity=gravity,
                                  coulomb_fric=cfric,
                                  inertia=inertia,
                                  motor_inertia=motor_inertia,
                                  torque_limit=torque_limit)
    controller.set_parameters(kp=kp, kd=kd, kv=kv)
    controller.set_goal(goal)
    # controller.check_parameters()

    states =[[0., 0., 0., 0.],
             [np.pi, 0., 0., 0.],
             [0., 1.0, -5, 12.],
             [5*np.pi, -3*np.pi, -1e-5, 100],
             [1e5, 1e-11, 1, -3],
             [1, 0, 0, 1],
            ]

    def test_0_get_control_output(self):
        self.controller.init()
        for x in self.states:
            u = self.controller.get_control_output(x)
            self.assertTrue(len(u) == 2)

    def test_1_acrobot(self):
        self.controller.init()
        T, X, U = self.sim.simulate(t0=0.0,
                                    x0=self.x0,
                                    tf=self.t_final,
                                    dt=self.dt,
                                    controller=self.controller,
                                    integrator=self.integrator)
