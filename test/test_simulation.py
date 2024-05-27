"""
Unit Tests
==========
"""

import unittest
import numpy as np


from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator


class Test(unittest.TestCase):
    plant = SymbolicDoublePendulum(
        mass=[0.606, 0.630],
        length=[0.3, 0.2],
        com=[0.275, 0.166],
        damping=[0.081, 0.0],
        gravity=9.81,
        coulomb_fric=[0.093, 0.186],
        inertia=[None, None],
        motor_inertia=0.0,
        gear_ratio=6,
        torque_limit=[3.0, 3.0],
    )

    simulator = Simulator(plant)

    states = [
        [0.0, 0.0, 0.0, 0.0],
        [np.pi, 0.0, 0.0, 0.0],
        [0.0, 1.0, -5, 12.0],
        [5 * np.pi, -3 * np.pi, -1e-5, 100],
        [1e5, 1e-11, 1, -3],
        [1, 0, 0, 1],
    ]

    actions = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-4.0, 5],
        [10, 10],
        [-10, 10],
        [1e-10, 0.5],
    ]

    times = [0.0, 1, 2.0]
    final_times = [10.0]
    dts = [0.001, 0.01, 0.1, 1.0]

    controllers = [None]
    # TODO: simulation with controllers, filters, noises, etc.

    def test_0_set_and_get_state(self):
        for t in self.times:
            for x in self.states:
                self.simulator.set_state(t, x)
                tt, xx = self.simulator.get_state()
                self.assertTrue(np.abs(t - tt) < 1e-5)
                self.assertTrue(np.max(np.abs(np.asarray(x) - np.asarray(xx))) < 1e-5)

    def test_1_reset_data_recorder(self):
        self.simulator.reset_data_recorder()
        self.assertTrue(self.simulator.t_values == [])
        self.assertTrue(self.simulator.x_values == [])
        self.assertTrue(self.simulator.tau_values == [])
        self.assertTrue(self.simulator.meas_x_values == [])
        # self.assertTrue(self.simulator.filt_x_values == [])
        self.assertTrue(self.simulator.con_u_values == [])

    def test_2_record_data(self):
        for t in self.times:
            for x in self.states:
                for a in self.actions:
                    self.simulator.record_data(t, x, a)
                    tt = self.simulator.t_values[-1]
                    xx = self.simulator.x_values[-1]
                    aa = self.simulator.tau_values[-1]
                    self.assertTrue(np.abs(t - tt) < 1e-5)
                    self.assertTrue(
                        np.max(np.abs(np.asarray(x) - np.asarray(xx))) < 1e-5
                    )
                    self.assertTrue(
                        np.max(np.abs(np.asarray(a) - np.asarray(aa))) < 1e-5
                    )
        self.simulator.reset_data_recorder()

    def test_3_get_trajectory_data(self):
        self.simulator.reset_data_recorder()
        N = 0
        for t in self.times:
            for x in self.states:
                for a in self.actions:
                    self.simulator.record_data(t, x, a)
                    N += 1

        T, X, U = self.simulator.get_trajectory_data()
        print("T", np.shape(T), N)
        self.assertTrue(np.shape(T) == (N,))
        self.assertTrue(np.shape(X) == (N, 4))
        self.assertTrue(np.shape(U) == (N, 2))

    def test_4_euler_integrator(self):
        for t in self.times:
            for x in self.states:
                for a in self.actions:
                    for dt in self.dts:
                        res = self.simulator.euler_integrator(x, dt, t, a)
                        self.assertTrue(type(res) == np.ndarray)
                        self.assertTrue(np.shape(res) == (4,))
                        self.assertTrue(not None in res)

    def test_4_runge_integrator(self):
        for t in self.times:
            for x in self.states:
                for a in self.actions:
                    for dt in self.dts:
                        res = self.simulator.runge_integrator(x, dt, t, a)
                        self.assertTrue(type(res) == np.ndarray)
                        self.assertTrue(np.shape(res) == (4,))
                        self.assertTrue(not None in res)

    def test_5_step(self):
        for integrator in ["euler", "runge_kutta"]:
            for dt in self.dts:
                self.simulator.reset()
                for a in self.actions:
                    self.simulator.step(a, dt, integrator)

    def test_6_simulate(self):
        for integrator in ["euler", "runge_kutta"]:
            for dt in self.dts:
                for t in self.times:
                    for tf in self.final_times:
                        for x in self.states:
                            for c in self.controllers:
                                T, X, U = self.simulator.simulate(
                                    t, x, tf, dt, c, integrator
                                )
                                N = int((tf - t) / dt + 1)
                                print((tf - t) / dt + 1)
                                print(N)
                                # N/N+1 ambiguity due to floating point errors
                                # in while(t < tf) loop
                                print(np.shape(T), np.shape(T) in [(N,), (N + 1,)])
                                print(np.shape(X))
                                print(np.shape(U))
                                print(x)
                                print()
                                self.assertTrue(np.shape(T) in [(N,), (N + 1,)])
                                self.assertTrue(np.shape(X) in [(N, 4), (N + 1, 4)])
                                self.assertTrue(np.shape(U) in [(N - 1, 2), (N, 2)])
