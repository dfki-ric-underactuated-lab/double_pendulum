import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController


class PointPIDController(AbstractController):
    def __init__(self,
                 torque_limit=[1.0, 1.0],
                 dt=0.01):

        self.torque_limit = torque_limit
        self.dt = dt

        # default weights
        self.Kp = 1.0
        self.Ki = 0.0
        self.Kd = 0.1
        self.goal = np.array([np.pi, 0., 0., 0.])

        # init pars
        self.errors1 = []
        self.errors2 = []

    def set_parameters(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def set_goal(self, x):
        self.goal = x

    def init_(self):
        self.errors1 = []
        self.errors2 = []

    def get_control_output_(self, x, t=None):
        e1 = self.goal[0] - x[0]
        e2 = self.goal[1] - x[1]
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

        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])
        u = np.asarray([u1, u2])
        return u
