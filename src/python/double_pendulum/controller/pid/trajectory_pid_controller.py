import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.pcw_polynomial import FitPiecewisePolynomial, InterpolateVector
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties


class TrajPIDController(AbstractController):
    def __init__(self,
                 T=None,
                 X=None,
                 U=None,
                 csv_path=None,
                 read_with="pandas",
                 keys="",
                 use_feed_forward_torque=True,
                 torque_limit=[0.0, 1.0],
                 num_break=40):

        self.use_ff = use_feed_forward_torque
        self.torque_limit = torque_limit

        # load trajectory
        if csv_path is not None:
            self.T, self.X, self.U = load_trajectory(
                    csv_path=csv_path,
                    read_with=read_with,
                    with_tau=self.use_ff,
                    keys=keys)
        elif T is not None and X is not None:
            self.T = T
            self.X = X
            if U is not None:
                self.U = U
            else:
                self.U = np.zeros((len(self.T), 2))
        else:
            print("Please Parse a trajectory to the TrajPIDController")
            exit()

        self.dt, self.max_t, _, _ = trajectory_properties(self.T, self.X)

        # interpolate trajectory
        self.P_interp = InterpolateVector(
                T=self.T,
                X=self.X.T[:2].T,
                num_break=num_break,
                poly_degree=3)

        if self.use_ff:
            self.U_interp = InterpolateVector(
                    T=self.T,
                    X=self.U,
                    num_break=num_break,
                    poly_degree=3)

        # default weights
        self.Kp = 10.0
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
        tt = min(t, self.max_t)

        p = self.P_interp.get_value(tt)
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

        if self.use_ff:
            uu = self.U_interp.get_value(tt)
            u1 = uu[0] + P1 + I1 + D1
            u2 = uu[1] + P2 + I2 + D2
        else:
            u1 = P1 + I1 + D1
            u2 = P2 + I2 + D2

        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])
        u = np.asarray([u1, u2])
        return u

    def get_init_trajectory(self):
        return self.T, self.X, self.U
