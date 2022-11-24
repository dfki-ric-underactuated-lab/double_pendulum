"""
Unit Tests
==========
"""

import unittest
import numpy as np


from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import SymbolicPFLController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator


class Test(unittest.TestCase):

    mpar = model_parameters()
    torque_limits = [[0., 5.], [5., 0.]]
    robots = ["pendubot", "acrobot"]
    pfl_methods = ["collocated", "noncollocated"]
    pars = [[0.0093613, 0.99787652, 0.9778557 ],
            [9.19534629, 2.24529733, 5.90567362],
            [8.8295605, 6.78718988, 4.42965278],
            [8.0722899, 4.92133648, 3.53211381]]

    goals = [[np.pi, 0., 0., 0.]]

    plant = SymbolicDoublePendulum(model_pars=mpar)
    sim = Simulator(plant=plant)

    states =[[0., 0., 0., 0.],
             [np.pi, 0., 0., 0.],
             [0., 1.0, -5, 12.],
             [5*np.pi, -3*np.pi, -1e-5, 100],
             [1e5, 1e-11, 1, -3],
             [1, 0, 0, 1],
            ]

    def test_0(self):
        for rob in self.robots:
            for pfl in self.pfl_methods:
                controller = SymbolicPFLController(model_pars=self.mpar,
                                                   robot=rob,
                                                   pfl_method=pfl)
                for g in self.goals:
                    for par in self.pars:
                        controller.set_goal(g)
                        controller.set_cost_parameters_(par)
                        controller.init()
                        for x in self.states:
                            u = controller.get_control_output(x)
                            self.assertTrue(len(u) == 2)
                            if rob == "pendubot":
                                self.assertTrue(u[1] <= 1e-5)
                            elif rob == "acrobot":
                                self.assertTrue(u[0] <= 1e-5)
