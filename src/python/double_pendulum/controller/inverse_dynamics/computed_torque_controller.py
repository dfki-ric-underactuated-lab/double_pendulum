import numpy as np
import pandas as pd

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


class ComputedTorqueController(AbstractController):
    def __init__(self,
                 mass=[0.5, 0.6],
                 length=[0.3, 0.2],
                 com=[0.3, 0.2],
                 damping=[0.1, 0.1],
                 coulomb_fric=[0.0, 0.0],
                 gravity=9.81,
                 inertia=[None, None],
                 torque_limit=[0.0, 1.0],
                 model_pars=None,
                 dt=0.01,
                 csv_path=None):

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.cfric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.cfric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            # self.Ir = model_pars.Ir
            # self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        self.splant = SymbolicDoublePendulum(mass=self.mass,
                                             length=self.length,
                                             com=self.com,
                                             damping=self.damping,
                                             gravity=self.gravity,
                                             coulomb_fric=self.cfric,
                                             inertia=self.inertia,
                                             torque_limit=self.torque_limit)
        self.dt = dt

        data = pd.read_csv(csv_path)
        time_traj = np.asarray(data["time"])
        pos1_traj = np.asarray(data["pos1"])
        pos2_traj = np.asarray(data["pos2"])
        vel1_traj = np.asarray(data["vel1"])
        vel2_traj = np.asarray(data["vel2"])
        acc1_traj = np.asarray(data["acc1"])
        acc2_traj = np.asarray(data["acc2"])

        self.T = time_traj.T
        self.X = np.asarray([pos1_traj, pos2_traj,
                             vel1_traj, vel2_traj]).T
        self.ACC = np.asarray([acc1_traj, acc2_traj]).T

        # default weights
        self.Kp = 10.0
        self.Ki = 0.0
        self.Kd = 0.1

        # init pars
        self.errors1 = []
        self.errors2 = []
        self.counter = 0

    def set_parameters(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def init_(self):
        self.errors1 = []
        self.errors2 = []
        self.counter = 0

    def get_control_output_(self, x, t=None):

        acc = self.splant.inverse_dynamics(self.X[self.counter], self.ACC[self.counter])

        p = self.X[self.counter, :2]

        e1 = p[0] - x[0]
        e2 = p[1] - x[1]
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

        u1 = P1 + I1 + D1
        u2 = P2 + I2 + D2
        u = np.asarray([u1, u2])

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        self.counter += 1
        return u

    def get_init_trajectory(self):
        return self.T, self.X, None 
