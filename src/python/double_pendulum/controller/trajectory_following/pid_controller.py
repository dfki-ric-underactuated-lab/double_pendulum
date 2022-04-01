import time
import numpy as np
import pandas as pd

from double_pendulum.controller.abstract_controller import AbstractController


class PIDController(AbstractController):
    def __init__(self,
                 csv_path,
                 read_with="pandas",
                 use_feed_forward_torque=True,
                 torque_limit=[0.0, 1.0]):

        self.use_ff = use_feed_forward_torque
        self.torque_limit = torque_limit

        if read_with == "pandas":
            self.data = pd.read_csv(csv_path)

            self.time_traj = np.asarray(self.data["time"])
            self.pos1_traj = np.asarray(self.data["shoulder_pos"])
            self.pos2_traj = np.asarray(self.data["elbow_pos"])
            self.vel1_traj = np.asarray(self.data["shoulder_vel"])
            self.vel2_traj = np.asarray(self.data["elbow_vel"])
            if self.use_ff:
                self.tau1_traj = np.asarray(self.data["shoulder_torque"])
                self.tau2_traj = np.asarray(self.data["elbow_torque"])

        elif read_with == "numpy":
            self.data = np.loadtxt(csv_path, skiprows=1, delimiter=",")

            self.time_traj = self.data[:,0]
            self.pos1_traj = self.data[:,1]
            self.pos2_traj = self.data[:,2]
            self.vel1_traj = self.data[:,3]
            self.vel1_traj = self.data[:,4]
            if self.use_ff:
                self.tau1_traj = self.data[:,5]
                self.tau2_traj = self.data[:,6]

        self.dt = self.time_traj[1] - self.time_traj[0]
        self.max_t = self.time_traj[-1]

        # default weights
        self.Kp = 1.0
        self.Ki = 0.0
        self.Kd = 0.1

        # init pars
        self.errors1 = []
        self.errors2 = []

    def set_parameters(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def init(self):
        self.errors1 = []
        self.errors2 = []

    def get_control_output(self, x, t):
        n = int(np.around(min(t, self.max_t) / self.dt))

        e1 = self.pos1_traj[n] - x[0]
        e2 = self.pos2_traj[n] - x[1]
        e1 = (e1 + np.pi) % (2*np.pi) - np.pi
        e2 = (e2 + np.pi) % (2*np.pi) - np.pi
        self.errors1.append(e1)
        self.errors2.append(e2)

        P1 = self.Kp*e1
        P2 = self.Kp*e2

        I1 = self.Ki*np.sum(np.asarray(self.errors1))*self.dt
        I2 = self.Ki*np.sum(np.asarray(self.errors2))*self.dt
        if len(self.errors1) > 2:
            D1 = self.Kd*(self.errors1[-1]-self.errors1[-2]) / self.dt
            D2 = self.Kd*(self.errors2[-1]-self.errors2[-2]) / self.dt
        else:
            D1 = 0.0
            D2 = 0.0

        if self.use_ff:
            u1 = self.tau1_traj[n] + P1 + I1 + D1
            u2 = self.tau2_traj[n] + P2 + I2 + D2
        else:
            u1 = P1 + I1 + D1
            u2 = P2 + I2 + D2

        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])
        u = np.asarray([u1, u2])
        return u

    def get_init_trajectory(self):
        T = self.time_traj.T
        X = np.asarray([self.pos1_traj, self.pos2_traj,
                        self.vel1_traj, self.vel2_traj]).T
        if self.use_ff:
            U = np.asarray([self.tau1_traj, self.tau2_traj]).T
        else:
            U = np.asarray([np.zeros_like(T), np.zeros_like(T)])
        return T, X, U
