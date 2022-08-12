import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.pcw_polynomial import InterpolateVector
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties, load_Kk_values


class TrajectoryController(AbstractController):
    def __init__(self,
                 csv_path,
                 read_with,
                 keys="",
                 torque_limit=[0.0, 1.0],
                 kK_stabilization=False):

        self.torque_limit = torque_limit
        self.kK_stabilization = kK_stabilization

        self.T, self.X, self.U = load_trajectory(csv_path, read_with, with_tau=True, keys=keys)
        self.dt, self.max_t, _, _ = trajectory_properties(self.T, self.X)
        if self.kK_stabilization:
            self.K1, self.K2, self.k1, self.k2 = load_Kk_values(csv_path, read_with, keys=keys)

    def get_control_output_(self, x, t):
        n = int(np.around(min(t, self.max_t) / self.dt))

        u1 = self.U[n][0]
        u2 = self.U[n][1]

        if self.kK_stabilization:
            x_des = self.X[n]

            K1 = self.K1[n]
            K2 = self.K2[n]

            # k1 = self.k1[n]
            # k2 = self.k2[n]
            k1 = 0.0
            k2 = 0.0

            u1 = u1 + k1 - np.dot(K1, x_des - x)
            u2 = u2 + k2 - np.dot(K2, x_des - x)
        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])

        u = np.asarray([u1, u2])
        return u

    def get_init_trajectory(self):
        return self.T, self.X, self.U


class TrajectoryInterpController(AbstractController):
    def __init__(self,
                 csv_path,
                 read_with="numpy",
                 keys="",
                 torque_limit=[0.0, 1.0],
                 kK_stabilization=False,
                 num_break=40):

        self.torque_limit = torque_limit
        self.kK_stabilization = kK_stabilization

        self.T, self.X, self.U = load_trajectory(csv_path, read_with, with_tau=True, keys=keys)
        self.dt, self.max_t, _, _ = trajectory_properties(self.T, self.X)
        if self.kK_stabilization:
            self.K1, self.K2, self.k1, self.k2 = load_Kk_values(csv_path, read_with, keys=keys)

        self.U_interp = InterpolateVector(
                T=self.T,
                X=self.U,
                num_break=num_break,
                poly_degree=3)

        if self.kK_stabilization:
            self.X_interp = InterpolateVector(
                    T=self.T,
                    X=self.X,
                    num_break=num_break,
                    poly_degree=3)
            self.K1_interp = InterpolateVector(
                    T=self.T,
                    X=self.K1,
                    num_break=num_break,
                    poly_degree=3)
            self.K2_interp = InterpolateVector(
                    T=self.T,
                    X=self.K2,
                    num_break=num_break,
                    poly_degree=3)
            k = np.swapaxes([self.k1, self.k2], 0, 1)
            self.k_interp = InterpolateVector(
                    T=self.T,
                    X=k,
                    num_break=num_break,
                    poly_degree=3)

    def get_control_output_(self, x, t):
        tt = min(t, self.max_t)

        uu = self.U_interp.get_value(tt)
        u1 = uu[0]
        u2 = uu[1]

        if self.kK_stabilization:
            x_des = self.X_interp.get_value(tt)

            K1 = self.K1_interp.get_value(tt)
            K2 = self.K2_interp.get_value(tt)

            # k = self.k_interp.get_value(tt)
            # k1 = k[0]
            # k2 = k[1]
            k1 = 0.0
            k2 = 0.0

            u1 = u1 + k1 - np.dot(K1, x_des - x)
            u2 = u2 + k2 - np.dot(K2, x_des - x)
        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])

        u = np.asarray([u1, u2])
        return u

    def get_init_trajectory(self):
        return self.T, self.X, self.U
