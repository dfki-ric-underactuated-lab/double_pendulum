import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.pcw_polynomial import FitPiecewisePolynomial, InterpolateVector


class FeedForwardController(AbstractController):
    def __init__(self,
                 T,
                 U,
                 torque_limit=[1.0, 1.0],
                 num_break=40):

        self.T = T
        self.U = U
        self.torque_limit = torque_limit
        self.num_break = num_break

        self.U_interp = InterpolateVector(
                T=self.T,
                X=self.U,
                num_break=num_break,
                poly_degree=3)

    def get_control_output(self, x, t):
        tt = min(t, self.T[-1])

        uu = self.U_interp.get_value(tt)
        u1 = np.clip(uu[0], -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(uu[1], -self.torque_limit[1], self.torque_limit[1])

        u = np.asarray([u1, u2])
        return u
