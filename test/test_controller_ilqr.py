"""
Unit Tests
==========
"""

import os
import unittest
import numpy as np


from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator


class Test(unittest.TestCase):

    mpar = model_parameters(mass=[0.64, 0.56],
                            length=[0.2, 0.3],
                            com=[0.2, 0.32],
                            damping=[0., 0.],
                            cfric=[0., 0.],
                            gravity=9.81,
                            inertia=[0.027, 0.054],
                            torque_limit=[0., 6.])

    # simulation parameter
    dt = 0.005
    t_final = 10.  # 5.985
    integrator = "runge_kutta"
    start = [0., 0., 0., 0.]
    goal = [np.pi, 0., 0., 0.]

    # controller parameters
    N = 100
    con_dt = dt
    N_init = 1000
    max_iter = 10
    max_iter_init = 1000
    regu_init = 1.
    max_regu = 10000.
    min_regu = 0.01
    break_cost_redu = 1e-6
    trajectory_stabilization = True
    shifting = 1

    sCu = [.1, .1]
    sCp = [.1, .1]
    sCv = [0.01, 0.1]
    sCen = 0.0
    fCp = [100., 10.]
    fCv = [10., 1.]
    fCen = 0.0

    f_sCu = [0.1, 0.1]
    f_sCp = [.1, .1]
    f_sCv = [.01, .01]
    f_sCen = 0.0
    f_fCp = [10., 10.]
    f_fCv = [1., 1.]
    f_fCen = 0.0

    controller = ILQRMPCCPPController(model_pars=mpar)
    controller.set_start(start)
    controller.set_goal(goal)
    controller.set_parameters(N=N,
                              dt=con_dt,
                              max_iter=max_iter,
                              regu_init=regu_init,
                              max_regu=max_regu,
                              min_regu=min_regu,
                              break_cost_redu=break_cost_redu,
                              integrator=integrator,
                              trajectory_stabilization=trajectory_stabilization,
                              shifting=shifting)
    controller.set_cost_parameters(sCu=sCu,
                                   sCp=sCp,
                                   sCv=sCv,
                                   sCen=sCen,
                                   fCp=fCp,
                                   fCv=fCv,
                                   fCen=fCen)
    controller.set_final_cost_parameters(sCu=f_sCu,
                                         sCp=f_sCp,
                                         sCv=f_sCv,
                                         sCen=f_sCen,
                                         fCp=f_fCp,
                                         fCv=f_fCv,
                                         fCen=f_fCen)

    states =[[0., 0., 0., 0.],
             [np.pi, 0., 0., 0.],
             [0., 1.0, -5, 12.],
             [5*np.pi, -3*np.pi, -1e-5, 100],
             [1e5, 1e-11, 1, -3],
             [1, 0, 0, 1],
            ]

    plant = SymbolicDoublePendulum(model_pars=mpar)
    sim = Simulator(plant=plant)

    def test_0_load_init_traj(self):
        init_csv_path = os.path.join(
                "../data/trajectories",
                "design_C.0",
                "model_3.1",
                "acrobot",
                "ilqr_2/trajectory.csv")
        self.controller.load_init_traj(csv_path=init_csv_path,
                                  num_break=40,
                                  poly_degree=3)
        self.controller.init()

    def test_1_compute_init_traj(self):
        self.controller.compute_init_traj()
        self.controller.init()

    def test_2_execute_controller(self):
        init_csv_path = os.path.join(
                "../data/trajectories",
                "design_C.0",
                "model_3.1",
                "acrobot",
                "ilqr_2/trajectory.csv")
        self.controller.load_init_traj(csv_path=init_csv_path,
                                  num_break=40,
                                  poly_degree=3)
        self.controller.init()
        T, X, U = self.sim.simulate(
                    t0=0.0,
                    x0=self.start,
                    tf=self.t_final,
                    dt=self.dt,
                    controller=self.controller,
                    integrator="runge_kutta")
        diff = np.max(np.abs(np.asarray(X[-1]) - np.asarray(self.goal)))
        self.assertTrue(diff < 1e-2)
        self.assertTrue(np.max(np.abs(U)) < self.mpar.tl[1])
