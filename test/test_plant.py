"""
Unit Tests
==========
"""

import unittest
import numpy as np


from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters


class Test(unittest.TestCase):

    plant1 = DoublePendulumPlant(mass=[0.606, 0.630],
                                 length=[0.3, 0.2],
                                 com=[0.275, 0.166],
                                 damping=[0.081, 0.0],
                                 gravity=9.81,
                                 coulomb_fric=[0.093, 0.186],
                                 inertia=[None, None],
                                 motor_inertia=0.0,
                                 gear_ratio=6,
                                 torque_limit=[3.0, 3.0])
    plant2 = DoublePendulumPlant(mass=[0.64, 0.56],
                                 length=[0.2, 0.3],
                                 com=[0.2, 0.32],
                                 damping=[0.001, 0.001],
                                 gravity=9.81,
                                 coulomb_fric=[0.078, 0.093],
                                 inertia=[0.027, 0.054],
                                 motor_inertia=6.29e-5,
                                 gear_ratio=6,
                                 torque_limit=[10.0, 10.0])

    mpar = model_parameters()
    plant3 = DoublePendulumPlant(model_pars=mpar)

    plants = [plant1, plant2, plant3]

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


    def test_0_forward_kinematics(self):

        for p in self.plants:
            for pos in self.poses:
                coord = p.forward_kinematics(pos)
                self.assertTrue(type(coord) == list)
                self.assertTrue(len(coord) == 2)
                self.assertTrue(len(coord[0]) == 2)
                self.assertTrue(len(coord[1]) == 2)
                self.assertTrue(not None in coord)

    def test_1_mass_matrix(self):

        for p in self.plants:
            for x in self.states:
                M = p.mass_matrix(x)
                self.assertTrue(type(M) == np.ndarray)
                self.assertTrue(np.shape(M) == (2,2))
                self.assertTrue(not None in M)

    def test_2_coriolis_matrix(self):

        for p in self.plants:
            for x in self.states:
                C = p.coriolis_matrix(x)
                self.assertTrue(type(C) == np.ndarray)
                self.assertTrue(np.shape(C) == (2,2))
                self.assertTrue(not None in C)

    def test_3_gravity_vector(self):

        for p in self.plants:
            for x in self.states:
                G = p.gravity_vector(x)
                self.assertTrue(type(G) == np.ndarray)
                self.assertTrue(np.shape(G) == (2,))
                self.assertTrue(not None in G)

    def test_4_coulomb_vector(self):

        for p in self.plants:
            for x in self.states:
                F = p.coulomb_vector(x)
                self.assertTrue(type(F) == np.ndarray)
                self.assertTrue(np.shape(F) == (2,))
                self.assertTrue(not None in F)

    def test_5_kinetic_energy(self):

        for p in self.plants:
            for x in self.states:
                E = p.kinetic_energy(x)
                self.assertTrue(type(E) in [float, int, np.float64])

    def test_6_potential_energy(self):

        for p in self.plants:
            for x in self.states:
                E = p.potential_energy(x)
                self.assertTrue(type(E) in [float, int, np.float64])

    def test_7_total_energy(self):

        for p in self.plants:
            for x in self.states:
                E = p.total_energy(x)
                self.assertTrue(type(E) in [float, int, np.float64])

    def test_8_forward_dynamics(self):

        for p in self.plants:
            for x in self.states:
                for u in self.actions:
                    accn = p.forward_dynamics(x, u)
                    self.assertTrue(type(accn) == np.ndarray)
                    self.assertTrue(np.shape(accn) == (2,))
                    self.assertTrue(not None in accn)

    def test_9_rhs(self):

        for p in self.plants:
            for t in self.times:
                for x in self.states:
                    for u in self.actions:
                        res = p.rhs(t, x, u)
                        self.assertTrue(type(res) == np.ndarray)
                        self.assertTrue(np.shape(res) == (4,))
                        self.assertTrue(not None in res)

    def test_10_Mx(self):

        for p in self.plants:
            for x in self.states:
                for u in self.actions:
                    Mx = p.get_Mx(x, u)
                    self.assertTrue(type(Mx) == np.ndarray)
                    self.assertTrue(np.shape(Mx) == (4,2,2))
                    self.assertTrue(not None in Mx)

    def test_11_Minvx(self):

        for p in self.plants:
            for x in self.states:
                for u in self.actions:
                    Minvx = p.get_Minvx(x, u)
                    self.assertTrue(type(Minvx) == np.ndarray)
                    self.assertTrue(np.shape(Minvx) == (4,2,2))
                    self.assertTrue(not None in Minvx)

    def test_12_Cx(self):

        for p in self.plants:
            for x in self.states:
                for u in self.actions:
                    Cx = p.get_Cx(x, u)
                    self.assertTrue(type(Cx) == np.ndarray)
                    self.assertTrue(np.shape(Cx) == (4,2,2))
                    self.assertTrue(not None in Cx)

    def test_13_Gx(self):

        for p in self.plants:
            for x in self.states:
                for u in self.actions:
                    Gx = p.get_Gx(x, u)
                    self.assertTrue(type(Gx) == np.ndarray)
                    self.assertTrue(np.shape(Gx) == (2,4))
                    self.assertTrue(not None in Gx)

    def test_14_Fx(self):

        for p in self.plants:
            for x in self.states:
                for u in self.actions:
                    Fx = p.get_Fx(x, u)
                    self.assertTrue(type(Fx) == np.ndarray)
                    self.assertTrue(np.shape(Fx) == (2,4))
                    self.assertTrue(not None in Fx)

    def test_15_Alin(self):

        for p in self.plants:
            for x in self.states:
                for u in self.actions:
                    A = p.get_Alin(x, u)
                    self.assertTrue(type(A) == np.ndarray)
                    self.assertTrue(np.shape(A) == (4,4))
                    self.assertTrue(not None in A)

    def test_16_Blin(self):

        for p in self.plants:
            for x in self.states:
                for u in self.actions:
                    B = p.get_Blin(x, u)
                    self.assertTrue(type(B) == np.ndarray)
                    self.assertTrue(np.shape(B) == (4,2))
                    self.assertTrue(not None in B)

    def test_17_linMats(self):

        for p in self.plants:
            for x in self.states:
                for u in self.actions:
                    A, B = p.linear_matrices(x, u)
                    self.assertTrue(type(A) == np.ndarray)
                    self.assertTrue(np.shape(A) == (4,4))
                    self.assertTrue(not None in A)
                    self.assertTrue(type(B) == np.ndarray)
                    self.assertTrue(np.shape(B) == (4,2))
                    self.assertTrue(not None in B)

if __name__ == '__main__':
    unittest.main()
