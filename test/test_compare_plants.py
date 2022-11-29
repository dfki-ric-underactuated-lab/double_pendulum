"""
Unit Tests
==========
"""

import unittest
import numpy as np


from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters


class Test(unittest.TestCase):

    plant1 = DoublePendulumPlant(
            mass=[0.606, 0.630],
            length=[0.3, 0.2],
            com=[0.275, 0.166],
            damping=[0.081, 0.0],
            gravity=9.81,
            coulomb_fric=[0.093, 0.186],
            inertia=[None, None],
            motor_inertia=0.0,
            gear_ratio=6,
            torque_limit=[3.0, 3.0])
    plant2 = DoublePendulumPlant(
            mass=[0.64, 0.56],
            length=[0.2, 0.3],
            com=[0.2, 0.32],
            damping=[0.001, 0.001],
            gravity=9.81,
            coulomb_fric=[0.078, 0.093],
            inertia=[0.027, 0.054],
            #motor_inertia=6.29e-5,  # derivatives in regular plant
            motor_inertia=0.,        # do not support Ir != 0
            gear_ratio=6,
            torque_limit=[10.0, 10.0])

    mpar = model_parameters()
    plant3 = DoublePendulumPlant(model_pars=mpar)

    plants = [plant1, plant2, plant3]

    splant1 = SymbolicDoublePendulum(
            mass=[0.606, 0.630],
            length=[0.3, 0.2],
            com=[0.275, 0.166],
            damping=[0.081, 0.0],
            gravity=9.81,
            coulomb_fric=[0.093, 0.186],
            inertia=[None, None],
            motor_inertia=0.0,
            gear_ratio=6,
            torque_limit=[3.0, 3.0])
    splant2 = SymbolicDoublePendulum(
            mass=[0.64, 0.56],
            length=[0.2, 0.3],
            com=[0.2, 0.32],
            damping=[0.001, 0.001],
            gravity=9.81,
            coulomb_fric=[0.078, 0.093],
            inertia=[0.027, 0.054],
            #motor_inertia=6.29e-5,
            motor_inertia=0.,
            gear_ratio=6,
            torque_limit=[10.0, 10.0])

    splant3 = SymbolicDoublePendulum(model_pars=mpar)

    splants = [splant1, splant2, splant3]

    poses = [[0., 0.],
             [np.pi, 0.],
             [0., 1.0],
             [5*np.pi, -3*np.pi],
             [1e5, -1],
             [1, 0],
            ]

    states =[[0., 0., 0., 0.],
             [np.pi, 0., 0., 0.],
             [0., 1.0, -5, 12.],
             [5*np.pi, -3*np.pi, -1e-5, 100],
             [1e5, 1e-11, 1, -3],
             [1, 0, 0, 1],
            ]

    actions = [[0.,0.],
               [1.0, 0.],
               [0., 1.],
               [-4., 5],
               [10, 10],
               [-10, 10],
               [1e-10, 0.5],
               ]
    times = [0., -1]

    epsilon = 1e-2

    def test_0_forward_kinematics(self):

        for i in range(len(self.plants)):
            for pos in self.poses:
                coord1 = self.plants[i].forward_kinematics(pos)
                coord2 = self.splants[i].forward_kinematics(pos)

                diff = np.max(np.abs(np.asarray(coord1) - np.asarray(coord2)))
                self.assertTrue(diff < self.epsilon)

    def test_1_mass_matrix(self):

        for i in range(len(self.plants)):
            for x in self.states:
                M1 = self.plants[i].mass_matrix(x)
                M2 = self.splants[i].mass_matrix(x)
                diff = np.max(np.abs(np.asarray(M1) - np.asarray(M2)))
                self.assertTrue(diff < self.epsilon)

    def test_2_coriolis_matrix(self):

        for i in range(len(self.plants)):
            for x in self.states:
                C1 = self.plants[i].coriolis_matrix(x)
                C2 = self.splants[i].coriolis_matrix(x)
                diff = np.max(np.abs(np.asarray(C1) - np.asarray(C2)))
                self.assertTrue(diff < self.epsilon)

    def test_3_gravity_vector(self):

        for i in range(len(self.plants)):
            for x in self.states:
                G1 = self.plants[i].gravity_vector(x)
                G2 = self.splants[i].gravity_vector(x)
                diff = np.max(np.abs(np.asarray(G1) - np.asarray(G2)))
                self.assertTrue(diff < self.epsilon)

    def test_4_coulomb_vector(self):

        for i in range(len(self.plants)):
            for x in self.states:
                F1 = self.plants[i].coulomb_vector(x)
                F2 = self.splants[i].coulomb_vector(x)
                diff = np.max(np.abs(np.asarray(F1) - np.asarray(F2)))
                self.assertTrue(diff < self.epsilon)

    def test_5_kinetic_energy(self):

        for i in range(len(self.plants)):
            for x in self.states:
                E1 = self.plants[i].kinetic_energy(x)
                E2 = self.splants[i].kinetic_energy(x)
                diff = np.max(np.abs(np.asarray(E1) - np.asarray(E2)))
                self.assertTrue(diff < self.epsilon)

    def test_6_potential_energy(self):

        for i in range(len(self.plants)):
            for x in self.states:
                E1 = self.plants[i].potential_energy(x)
                E2 = self.splants[i].potential_energy(x)
                diff = np.max(np.abs(np.asarray(E1) - np.asarray(E2)))
                self.assertTrue(diff < self.epsilon)

    def test_7_total_energy(self):

        for i in range(len(self.plants)):
            for x in self.states:
                E1 = self.plants[i].total_energy(x)
                E2 = self.splants[i].total_energy(x)
                diff = np.max(np.abs(np.asarray(E1) - np.asarray(E2)))
                self.assertTrue(diff < self.epsilon)

    def test_8_forward_dynamics(self):

        for i in range(len(self.plants)):
            for x in self.states:
                for u in self.actions:
                    accn1 = self.plants[i].forward_dynamics(x, u)
                    accn2 = self.splants[i].forward_dynamics(x, u)
                    diff = np.max(np.abs(np.asarray(accn1) - np.asarray(accn2)))
                    self.assertTrue(diff < self.epsilon)

    def test_9_rhs(self):

        for i in range(len(self.plants)):
            for t in self.times:
                for x in self.states:
                    for u in self.actions:
                        res1 = self.plants[i].rhs(t, x, u)
                        res2 = self.splants[i].rhs(t, x, u)
                        diff = np.max(np.abs(np.asarray(res1) - np.asarray(res2)))
                        self.assertTrue(diff < self.epsilon)

    def test_10_linMats(self):

        for i in range(len(self.plants)):
            for x in self.states:
                for u in self.actions:
                    A1, B1 = self.plants[i].linear_matrices(x, u)
                    A2, B2 = self.splants[i].linear_matrices(x, u)
                    diffA = np.max(np.abs(np.asarray(A1) - np.asarray(A2)))
                    diffB = np.max(np.abs(np.asarray(B1) - np.asarray(B2)))
                    if not diffA < self.epsilon:
                        print(i)
                        print("x", x, "u", u)
                        print("A1", A1, "B1", B1)
                        print("A2", A2, "B2", B2)
                        print("diffA", diffA)
                        print("diffB", diffB)
                    self.assertTrue(diffA < self.epsilon)
                    self.assertTrue(diffB < self.epsilon)

if __name__ == '__main__':
    unittest.main()
