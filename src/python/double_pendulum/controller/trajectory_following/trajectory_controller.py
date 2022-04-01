import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController


class TrajectoryController(AbstractController):
    def __init__(self,
                 csv_path,
                 torque_limit=[0.0, 1.0],
                 kK_stabilization=False):

        self.trajectory = np.loadtxt(csv_path, skiprows=1, delimiter=",")
        self.dt = self.trajectory[1][0] - self.trajectory[0][0]
        self.max_t = self.trajectory[-1][0]
        self.torque_limit = torque_limit

        self.kK_stabilization = kK_stabilization
        if self.kK_stabilization:
            if len(self.trajectory[0]) < 16:
                self.kK_stabilization = False
                print("Disabling kK_stabilization. No k/K terms found in csv file")

    def get_control_output(self, x, t):
        n = int(np.around(min(t, self.max_t) / self.dt))

        u1 = self.trajectory[n][5]
        u2 = self.trajectory[n][6]

        if self.kK_stabilization:
            x1_des = self.trajectory[n][1]
            x2_des = self.trajectory[n][2]
            x3_des = self.trajectory[n][3]
            x4_des = self.trajectory[n][4]
            x_des = np.asarray([x1_des, x2_des, x3_des, x4_des])

            K11 = self.trajectory[n][7]
            K12 = self.trajectory[n][8]
            K13 = self.trajectory[n][9]
            K14 = self.trajectory[n][10]
            K21 = self.trajectory[n][11]
            K22 = self.trajectory[n][12]
            K23 = self.trajectory[n][13]
            K24 = self.trajectory[n][14]
            K1 = np.asarray([K11, K12, K13, K14])
            K2 = np.asarray([K21, K22, K23, K24])

            # k1 = self.trajectory[n][15]
            # k2 = self.trajectory[n][16]
            k1 = 0.0
            k2 = 0.0

            u1 = u1 + k1 - np.dot(K1, x_des - x)
            u2 = u2 + k2 - np.dot(K2, x_des - x)
        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])

        u = np.asarray([u1, u2])
        return u

    def get_init_trajectory(self):
        u1_traj = self.trajectory.T[5]
        u2_traj = self.trajectory.T[6]
        p1_traj = self.trajectory.T[1]
        p2_traj = self.trajectory.T[2]
        v1_traj = self.trajectory.T[3]
        v2_traj = self.trajectory.T[4]

        T = self.trajectory.T[0]
        X = np.asarray([p1_traj, p2_traj, v1_traj, v2_traj]).T
        U = np.asarray([u1_traj, u2_traj]).T

        return T, X, U
