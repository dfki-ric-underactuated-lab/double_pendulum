"""
Unit Tests
==========
"""

import os
import unittest
import numpy as np


from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.tvlqr.tvlqr_controller_drake import TVLQRController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator


class Test(unittest.TestCase):

    mpar = model_parameters(model_design="design_C.0",
                            model_id="model_3.1")

    goals = [[np.pi, 0., 0., 0.]]

    plant = SymbolicDoublePendulum(model_pars=mpar)
    sim = Simulator(plant=plant)

    Qs = [np.diag([0.64, 0.56, 0.13, 0.037])]
    Rs = [np.eye(1)*0.82]
    Qfs = [np.diag([0.64, 0.56, 0.13, 0.037])]


    states =[[0., 0., 0., 0.],
             [np.pi, 0., 0., 0.],
             [0., 1.0, -5, 12.],
             [5*np.pi, -3*np.pi, -1e-5, 100],
             [1e5, 1e-11, 1, -3],
             [1, 0, 0, 1],
            ]

    times = [0., 0.1, 0.99, 10., 1e4]

    def test_0_acrobot(self):
        csv_path = os.path.join("../data/trajectories",
                                "design_A.0",
                                "model_2.1",
                                "acrobot",
                                "ilqr_1/trajectory.csv")
        urdf_path = "../data/urdfs/design_A.0/model_1.0/acrobot.urdf"
        mpar = model_parameters(model_design="design_A.0",
                                model_id="model_2.1",
                                robot="acrobot")
        controller = TVLQRController(
                csv_path=csv_path,
                urdf_path=urdf_path,
                model_pars=mpar,
                torque_limit=mpar.tl,
                robot="acrobot",
                save_dir=".")

        for g in self.goals:
            for Q in self.Qs:
                for R in self.Rs:
                    for Qf in self.Qfs:
                        controller.set_goal(g)
                        controller.set_cost_parameters(Q=Q, R=R, Qf=Qf)
                        controller.init()
                        for x in self.states:
                            for t in self.times:
                                u = controller.get_control_output(x, t)
                                self.assertTrue(len(u) == 2)
                                self.assertTrue(np.abs(u[0]) <= mpar.tl[0])
                                self.assertTrue(np.abs(u[1]) <= mpar.tl[1])

    def test_1_pendubot(self):
        csv_path = os.path.join("../data/trajectories",
                                "design_A.0",
                                "model_2.1",
                                "pendubot",
                                "ilqr_1/trajectory.csv")
        urdf_path = "../data/urdfs/design_A.0/model_1.0/pendubot.urdf"
        mpar = model_parameters(model_design="design_A.0",
                                model_id="model_2.1",
                                robot="pendubot")
        controller = TVLQRController(
                csv_path=csv_path,
                urdf_path=urdf_path,
                model_pars=mpar,
                torque_limit=mpar.tl,
                robot="pendubot",
                save_dir=".")
        for g in self.goals:
            for Q in self.Qs:
                for R in self.Rs:
                    for Qf in self.Qfs:
                        controller.set_goal(g)
                        controller.set_cost_parameters(Q=Q, R=R, Qf=Qf)
                        controller.init()
                        for x in self.states:
                            for t in self.times:
                                u = controller.get_control_output(x, t)
                                self.assertTrue(len(u) == 2)
                                self.assertTrue(np.abs(u[0]) <= mpar.tl[0])
                                self.assertTrue(np.abs(u[1]) <= mpar.tl[1])
