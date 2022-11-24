"""
Unit Tests
==========
"""

import unittest
import numpy as np


from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator


class Test(unittest.TestCase):


    mpar = model_parameters()
    torque_limits = [[5., 5.], [0., 5.], [5., 0.]]

    goals = [[np.pi, 0., 0., 0.]]

    Qs = [np.diag([0.97, 0.93, 0.39, 0.26]),
          np.diag([0.00125, 0.65, 0.000688, 0.000936])]
    Rs = [np.diag((0.11, 0.11)), np.diag([25.0, 25.0])]

    states =[[0., 0., 0., 0.],
             [np.pi, 0., 0., 0.],
             [0., 1.0, -5, 12.],
             [5*np.pi, -3*np.pi, -1e-5, 100],
             [1e5, 1e-11, 1, -3],
             [1, 0, 0, 1],
            ]

    def test_0_double_pendulum(self):
        self.mpar.set_torque_limit(self.torque_limits[0])
        controller = LQRController(model_pars=self.mpar)
        for g in self.goals:
            for Q in self.Qs:
                for R in self.Rs:
                    controller.set_goal(g)
                    controller.set_cost_matrices(Q, R)
                    controller.init()
                    for x in self.states:
                        u = controller.get_control_output(x)
                        self.assertTrue(len(u) == 2)

    def test_1_acrobot(self):
        self.mpar.set_torque_limit(self.torque_limits[1])
        controller = LQRController(model_pars=self.mpar)
        controller.set_parameters(failure_value=0.,
                                  cost_to_go_cut=15.)
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

    def test_2_pendubot(self):
        self.mpar.set_torque_limit(self.torque_limits[2])
        controller = LQRController(model_pars=self.mpar)
        controller.set_parameters(failure_value=0.,
                                  cost_to_go_cut=15.)
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

    def test_3_acrobot_stab(self):
        design = "design_A.0"
        model = "model_2.0"

        torque_limit = [0.0, 5.0]

        model_par_path = "../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
        mpar = model_parameters(filepath=model_par_path)

        mpar.set_motor_inertia(0.)
        mpar.set_damping([0., 0.])
        mpar.set_cfric([0., 0.])
        mpar.set_torque_limit(torque_limit)

        # simulation parameters
        dt = 0.002
        t_final = 10.0
        integrator = "runge_kutta"
        goal = [np.pi, 0., 0., 0.]

        x0 = [np.pi+0.1, -0.1, -.5, 0.5]

        Q = np.diag((0.97, 0.93, 0.39, 0.26))
        R = np.diag((0.11, 0.11))

        plant = SymbolicDoublePendulum(model_pars=mpar)

        sim = Simulator(plant=plant)

        controller = LQRController(model_pars=mpar)
        controller.set_goal(goal)
        controller.set_cost_matrices(Q=Q, R=R)
        controller.set_parameters(failure_value=0.0,
                                  cost_to_go_cut=1000)
        controller.init()
        T, X, U = sim.simulate(t0=0.0, x0=x0,
                               tf=t_final, dt=dt, controller=controller,
                               integrator=integrator)
        diff = np.max(np.abs(np.asarray(X[-1]) - np.asarray(goal)))
        self.assertTrue(diff < 1e-5)
        self.assertTrue(np.max(np.abs(U)) < torque_limit[1])

    def test_4_pendubot_stab(self):
        design = "design_A.0"
        model = "model_2.0"

        torque_limit = [5.0, 0.0]

        model_par_path = "../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
        mpar = model_parameters(filepath=model_par_path)

        mpar.set_motor_inertia(0.)
        mpar.set_damping([0., 0.])
        mpar.set_cfric([0., 0.])
        mpar.set_torque_limit(torque_limit)

        # simulation parameters
        dt = 0.002
        t_final = 10.0
        integrator = "runge_kutta"
        goal = [np.pi, 0., 0., 0.]

        x0 = [np.pi-0.2, 0.3, 0., 0.]

        Q = np.diag([0.00125, 0.65, 0.000688, 0.000936])
        R = np.diag([25.0, 25.0])

        plant = SymbolicDoublePendulum(model_pars=mpar)

        sim = Simulator(plant=plant)

        controller = LQRController(model_pars=mpar)
        controller.set_goal(goal)
        controller.set_cost_matrices(Q=Q, R=R)
        controller.set_parameters(failure_value=0.0,
                                  cost_to_go_cut=1000)
        controller.init()
        T, X, U = sim.simulate(t0=0.0, x0=x0,
                               tf=t_final, dt=dt, controller=controller,
                               integrator=integrator)
        diff = np.max(np.abs(np.asarray(X[-1]) - np.asarray(goal)))
        self.assertTrue(diff < 1e-5)
        self.assertTrue(np.max(np.abs(U)) < torque_limit[0])
