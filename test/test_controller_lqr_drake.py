"""
Unit Tests
==========
"""

import unittest
import numpy as np


from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.lqr_controller_drake import LQRController


class Test(unittest.TestCase):


    mpar = model_parameters()
    torque_limits = [[0., 5.], [5., 0.]]
    robots = ["acrobot", "pendubot"]

    goals = [[np.pi, 0., 0., 0.]]

    Qs = [np.diag([0.97, 0.93, 0.39, 0.26]),
          np.diag([0.00125, 0.65, 0.000688, 0.000936])]
    Rs = [np.eye(1)*0.11, np.eye(1)*25.]

    states =[[0., 0., 0., 0.],
             [np.pi, 0., 0., 0.],
             [0., 1.0, -5, 12.],
             [5*np.pi, -3*np.pi, -1e-5, 100],
             [1e5, 1e-11, 1, -3],
             [1, 0, 0, 1],
            ]

    def test_0_acrobot(self):
        urdf_path = "../data/urdfs/design_A.0/model_1.0/"+self.robots[0]+".urdf"
        self.mpar.set_torque_limit(self.torque_limits[0])
        controller = LQRController(
                urdf_path=urdf_path,
                model_pars=self.mpar,
                robot=self.robots[0],
                torque_limit=self.torque_limits[0])
        for g in self.goals:
            for Q in self.Qs:
                for R in self.Rs:
                    controller.set_goal(g)
                    controller.set_cost_matrices(Q, R)
                    controller.init()
                    for x in self.states:
                        u = controller.get_control_output(x)
                        self.assertTrue(len(u) == 2)
                        self.assertTrue(u[0] <= 1e-5)

    def test_1_pendubot(self):
        urdf_path = "../data/urdfs/design_A.0/model_1.0/"+self.robots[1]+".urdf"
        self.mpar.set_torque_limit(self.torque_limits[1])
        controller = LQRController(
                urdf_path=urdf_path,
                model_pars=self.mpar,
                robot=self.robots[1],
                torque_limit=self.torque_limits[1])
        for g in self.goals:
            for Q in self.Qs:
                for R in self.Rs:
                    controller.set_goal(g)
                    controller.set_cost_matrices(Q, R)
                    controller.init()
                    for x in self.states:
                        u = controller.get_control_output(x)
                        self.assertTrue(len(u) == 2)
                        self.assertTrue(u[1] <= 1e-5)

